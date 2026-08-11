use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use enigo::{Button, Coordinate, Direction, Enigo, Keyboard, Mouse, Settings};
use image::{imageops::FilterType, DynamicImage, ImageFormat};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    env,
    io::{BufRead, BufReader, Cursor},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::Duration,
};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;
use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_CONTROL, VK_ESCAPE, VK_MENU,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetForegroundWindow, GetWindowTextW, IsWindowVisible, SetForegroundWindow,
    ShowWindow, SW_RESTORE,
};
use xcap::Monitor;

const PROTOCOL_VERSION: &str = "1.0.0";
const LLAMA_ENDPOINT: &str = "http://127.0.0.1:8765/v1/chat/completions";

struct RuntimeState {
    snapshot: Mutex<RuntimeSnapshot>,
}

struct ModelState {
    process: Mutex<Option<Child>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelConfig {
    server_path: String,
    model_path: String,
    mmproj_path: String,
    llama_url: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            server_path: r"D:\llama\llama-server.exe".into(),
            model_path: r"D:\Projects\baodou\model\Qwen3.5-2B-UD-Q4_K_XL.gguf".into(),
            mmproj_path: r"D:\Projects\baodou\model\mmproj-F16.gguf".into(),
            llama_url: LLAMA_ENDPOINT.into(),
        }
    }
}

impl Default for ModelState {
    fn default() -> Self {
        Self {
            process: Mutex::new(None),
        }
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            snapshot: Mutex::new(RuntimeSnapshot::default()),
        }
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RuntimeSnapshot {
    protocol_version: String,
    mode: String,
    phase: String,
    connected: bool,
    inference_backend: String,
    device: String,
    model_ready: bool,
    task_id: Option<String>,
    goal: Option<String>,
    message: String,
}

impl Default for RuntimeSnapshot {
    fn default() -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION.into(),
            mode: "native computer use".into(),
            phase: "idle".into(),
            connected: true,
            inference_backend: "Rust planner / llama.cpp optional".into(),
            device: "SYCL0 · Intel Arc".into(),
            model_ready: false,
            task_id: None,
            goal: None,
            message: "Rust 本地运行时已就绪；不依赖 Python".into(),
        }
    }
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TaskRequest {
    goal: String,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct TaskEvent {
    task_id: String,
    phase: String,
    title: String,
    detail: String,
    timestamp: String,
    requires_confirmation: bool,
    complete: bool,
    ok: bool,
    raw: Option<Value>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PlanStep {
    action: String,
    target: String,
    x: Option<i32>,
    y: Option<i32>,
    text: Option<String>,
    #[serde(default)]
    risk: String,
    #[serde(default)]
    expected_change: String,
    #[serde(default)]
    requires_confirmation: bool,
}

#[derive(Clone)]
struct NativePlan {
    observation: String,
    summary: String,
    done: bool,
    step: Option<PlanStep>,
}

#[derive(Clone)]
struct ScreenFrame {
    png_base64: String,
    width: u32,
    height: u32,
    model_width: u32,
    model_height: u32,
}

fn emit(app: &AppHandle, event: TaskEvent) {
    let _ = app.emit("task-event", event);
}

fn update_snapshot(
    state: &RuntimeState,
    update: impl FnOnce(&mut RuntimeSnapshot),
) -> RuntimeSnapshot {
    let mut snapshot = state.snapshot.lock().expect("runtime state poisoned");
    update(&mut snapshot);
    snapshot.clone()
}

#[cfg(any())]
fn llama_endpoint() -> String {
    return LLAMA_ENDPOINT.into();
    let _default_endpoint = LLAMA_ENDPOINT;
    if env::var_os("BAODOU_LLAMA_URL").is_none() {
        return "http://127.0.0.1:8765/v1/chat/completions".into();
    }
    env::var("BAODOU_LLAMA_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8765/v1/chat/completions".into())
}

fn config_file(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    let directory = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("无法定位应用数据目录：{error}"))?;
    std::fs::create_dir_all(&directory)
        .map_err(|error| format!("无法创建应用数据目录：{error}"))?;
    Ok(directory.join("model-config.json"))
}

fn load_model_config(app: &AppHandle) -> ModelConfig {
    let config: ModelConfig = config_file(app)
        .ok()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default();
    if std::path::Path::new(&config.server_path).is_absolute()
        && std::path::Path::new(&config.model_path).is_absolute()
        && std::path::Path::new(&config.mmproj_path).is_absolute()
    {
        config
    } else {
        ModelConfig::default()
    }
}

#[tauri::command]
fn get_model_config(app: AppHandle) -> ModelConfig {
    load_model_config(&app)
}

#[tauri::command]
fn set_model_config(app: AppHandle, config: ModelConfig) -> Result<ModelConfig, String> {
    if config.server_path.trim().is_empty()
        || config.model_path.trim().is_empty()
        || config.mmproj_path.trim().is_empty()
        || config.llama_url.trim().is_empty()
    {
        return Err("模型程序、模型文件、MMPROJ 和接口 URL 都不能为空".into());
    }
    if !std::path::Path::new(config.server_path.trim()).is_absolute()
        || !std::path::Path::new(config.model_path.trim()).is_absolute()
        || !std::path::Path::new(config.mmproj_path.trim()).is_absolute()
    {
        return Err("llama-server、模型和 MMPROJ 路径必须使用绝对路径".into());
    }
    let normalized = ModelConfig {
        server_path: config.server_path.trim().into(),
        model_path: config.model_path.trim().into(),
        mmproj_path: config.mmproj_path.trim().into(),
        llama_url: config.llama_url.trim().into(),
    };
    let path = config_file(&app)?;
    let content = serde_json::to_string_pretty(&normalized)
        .map_err(|error| format!("配置序列化失败：{error}"))?;
    std::fs::write(path, content).map_err(|error| format!("配置保存失败：{error}"))?;
    stop_model(&app);
    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.model_ready = false;
        snapshot.inference_backend = "llama-server · 正在按新配置重启".into();
        snapshot.message = "配置已保存，正在停止旧模型并启动新模型…".into();
    });
    let handle = app.clone();
    std::thread::spawn(move || start_model(handle));
    Ok(normalized)
}

fn stop_model(app: &AppHandle) {
    let model_state = app.state::<ModelState>();
    let mut process = model_state.process.lock().expect("model state poisoned");
    if let Some(mut child) = process.take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn llama_health(endpoint: &str) -> bool {
    let health_url = endpoint
        .split("/v1/")
        .next()
        .map(|base| format!("{base}/health"))
        .unwrap_or_else(|| "http://127.0.0.1:8765/health".into());
    Client::builder()
        .timeout(Duration::from_millis(700))
        .build()
        .ok()
        .and_then(|client| client.get(health_url).send().ok())
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn start_model(app: AppHandle) {
    let config = load_model_config(&app);
    let endpoint = config.llama_url.clone();
    if llama_health(&endpoint) {
        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
            snapshot.model_ready = true;
            snapshot.inference_backend = "llama-server · 本地视觉模型".into();
            snapshot.message = "本地模型已连接".into();
        });
        return;
    }

    let server = std::path::PathBuf::from(&config.server_path);
    let model = std::path::PathBuf::from(&config.model_path);
    let mmproj = std::path::PathBuf::from(&config.mmproj_path);

    if !server.exists() || !model.exists() || !mmproj.exists() {
        let missing = if !server.exists() {
            server.display().to_string()
        } else if !model.exists() {
            model.display().to_string()
        } else {
            mmproj.display().to_string()
        };
        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
            snapshot.model_ready = false;
            snapshot.inference_backend = "llama-server · 未启动".into();
            snapshot.message = format!("模型启动失败，找不到：{missing}");
        });
        return;
    }

    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.message = "正在启动本地视觉模型…".into();
        snapshot.inference_backend = "llama-server · 启动中".into();
    });
    let port = endpoint
        .split(':')
        .last()
        .and_then(|value| value.split('/').next())
        .unwrap_or("8765");
    let host = endpoint
        .split("://")
        .nth(1)
        .and_then(|value| value.split('/').next())
        .and_then(|value| value.split(':').next())
        .unwrap_or("[IP]");
    let log_path = config_file(&app)
        .ok()
        .map(|path| path.with_file_name("llama-server.log"));
    let log_file = log_path.and_then(|path| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .ok()
    });
    let log_error = log_file.as_ref().and_then(|file| file.try_clone().ok());
    let child = Command::new(&server)
        .current_dir(server.parent().unwrap_or(std::path::Path::new(".")))
        .args(["--host", host])
        .args([
            "-m",
            model.to_string_lossy().as_ref(),
            "--mmproj",
            mmproj.to_string_lossy().as_ref(),
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--jinja",
        ])
        .stdin(Stdio::null())
        .stdout(log_file.map(Stdio::from).unwrap_or_else(Stdio::null))
        .stderr(log_error.map(Stdio::from).unwrap_or_else(Stdio::null))
        .spawn();
    match child {
        Ok(child) => {
            *app.state::<ModelState>()
                .process
                .lock()
                .expect("model state poisoned") = Some(child);
            for _ in 0..90 {
                if llama_health(&endpoint) {
                    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                        snapshot.model_ready = true;
                        snapshot.inference_backend = "llama-server · 本地视觉模型".into();
                        snapshot.message = "本地模型已就绪".into();
                    });
                    return;
                }
                if let Some(process) = app
                    .state::<ModelState>()
                    .process
                    .lock()
                    .expect("model state poisoned")
                    .as_mut()
                {
                    if let Ok(Some(status)) = process.try_wait() {
                        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                            snapshot.model_ready = false;
                            snapshot.inference_backend = "llama-server · 进程已退出".into();
                            snapshot.message = format!(
                                "模型进程启动后立即退出（{status}），详情见 llama-server.log"
                            );
                        });
                        return;
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
            update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                snapshot.model_ready = false;
                snapshot.inference_backend = "llama-server · 启动超时".into();
                snapshot.message = "模型进程已启动，但健康检查超时".into();
            });
        }
        Err(error) => {
            update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                snapshot.model_ready = false;
                snapshot.inference_backend = "llama-server · 启动失败".into();
                snapshot.message = format!("模型启动失败：{error}");
            });
        }
    }
}

fn event(
    task_id: &str,
    phase: &str,
    title: impl Into<String>,
    detail: impl Into<String>,
    confirmation: bool,
    complete: bool,
    ok: bool,
    raw: Option<Value>,
) -> TaskEvent {
    TaskEvent {
        task_id: task_id.into(),
        phase: phase.into(),
        title: title.into(),
        detail: detail.into(),
        timestamp: now(),
        requires_confirmation: confirmation,
        complete,
        ok,
        raw,
    }
}

#[tauri::command]
fn get_runtime(state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    state
        .snapshot
        .lock()
        .expect("runtime state poisoned")
        .clone()
}

#[tauri::command]
fn pause_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    let snapshot = update_snapshot(&state, |s| {
        s.phase = "paused".into();
        s.message = "已暂停；当前计划保留，等待继续或停止".into();
    });
    emit(
        &app,
        event(
            snapshot.task_id.as_deref().unwrap_or_default(),
            "paused",
            "任务已暂停",
            snapshot.message.clone(),
            false,
            false,
            true,
            None,
        ),
    );
    snapshot
}

#[tauri::command]
fn stop_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    let snapshot = update_snapshot(&state, |s| {
        s.phase = "stopped".into();
        s.message = "已停止当前任务，Rust 执行器未发送后续输入".into();
    });
    emit(
        &app,
        event(
            snapshot.task_id.as_deref().unwrap_or_default(),
            "stopped",
            "任务已停止",
            snapshot.message.clone(),
            false,
            true,
            true,
            None,
        ),
    );
    snapshot
}

fn start_emergency_stop_monitor(app: AppHandle) {
    std::thread::spawn(move || {
        let mut latched = false;
        loop {
            let pressed = unsafe {
                GetAsyncKeyState(VK_CONTROL.0 as i32) < 0
                    && GetAsyncKeyState(VK_MENU.0 as i32) < 0
                    && GetAsyncKeyState(VK_ESCAPE.0 as i32) < 0
            };
            if pressed && !latched {
                let state = app.state::<RuntimeState>();
                let snapshot = state
                    .snapshot
                    .lock()
                    .expect("runtime snapshot poisoned")
                    .clone();
                if ["observing", "planning", "executing"].contains(&snapshot.phase.as_str()) {
                    let stopped = update_snapshot(&state, |runtime| {
                        runtime.phase = "stopped".into();
                        runtime.message = "已通过全局快捷键中止 Computer Use".into();
                    });
                    emit(
                        &app,
                        event(
                            stopped.task_id.as_deref().unwrap_or_default(),
                            "stopped",
                            "全局紧急停止",
                            stopped.message,
                            false,
                            true,
                            true,
                            None,
                        ),
                    );
                }
                latched = true;
            } else if !pressed {
                latched = false;
            }
            std::thread::sleep(Duration::from_millis(40));
        }
    });
}

#[tauri::command]
fn run_task(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    request: TaskRequest,
) -> Result<String, String> {
    let goal = request.goal.trim().to_string();
    if goal.is_empty() {
        return Err("任务不能为空".into());
    }
    let task_id = Uuid::new_v4().to_string();
    let snapshot = update_snapshot(&state, |s| {
        s.mode = "native computer use".into();
        s.phase = "observing".into();
        s.task_id = Some(task_id.clone());
        s.goal = Some(goal.clone());
        s.message = "Rust 正在采集主屏幕并建立任务上下文".into();
    });
    emit(
        &app,
        event(
            &task_id,
            "observing",
            "正在观察桌面",
            snapshot.message,
            false,
            false,
            true,
            None,
        ),
    );

    let app_clone = app.clone();
    std::thread::spawn(move || run_native_task(app_clone, task_id, goal));
    Ok(snapshot.task_id.unwrap_or_default())
}

fn run_native_task(app: AppHandle, task_id: String, goal: String) {
    const MAX_STEPS: usize = 12;
    let endpoint = load_model_config(&app).llama_url;
    let mut history: Vec<String> = Vec::new();

    for step_index in 1..=MAX_STEPS {
        if !task_is_active(&app, &task_id) {
            return;
        }

        update_snapshot(&app.state::<RuntimeState>(), |s| {
            s.phase = "observing".into();
            s.message = format!("第 {step_index} 轮：正在观察目标应用");
        });
        emit(
            &app,
            event(
                &task_id,
                "observing",
                format!("第 {step_index} 轮 · 观察屏幕"),
                "正在获取最新屏幕状态并验证上一步结果",
                false,
                false,
                true,
                None,
            ),
        );

        let frame = match capture_primary() {
            Ok(frame) => frame,
            Err(error) => {
                finish_with_error(&app, &task_id, "屏幕采集失败", error);
                return;
            }
        };

        update_snapshot(&app.state::<RuntimeState>(), |s| {
            s.phase = "planning".into();
            s.message = format!("第 {step_index} 轮：正在决定下一步");
        });
        let visible_windows = visible_window_titles().unwrap_or_default();
        let foreground_window = foreground_window_title().unwrap_or_else(|| "Unknown".into());
        let foreground_handle = unsafe { GetForegroundWindow().0 };
        let plan = match infer_plan(
            &app,
            &task_id,
            &goal,
            &history,
            &visible_windows,
            &foreground_window,
            step_index,
            &frame,
            &endpoint,
        ) {
            Ok(plan) => plan,
            Err(error) => {
                if step_index == 1 && error.contains("下一步动作") {
                    match infer_window_intent(&goal, &visible_windows, &endpoint) {
                        Ok(Some(plan)) => plan,
                        Ok(None) => {
                            finish_with_error(&app, &task_id, "模型规划失败", error);
                            return;
                        }
                        Err(intent_error) => {
                            finish_with_error(
                                &app,
                                &task_id,
                                "窗口意图判断失败",
                                format!("{error}；{intent_error}"),
                            );
                            return;
                        }
                    }
                } else {
                    finish_with_error(&app, &task_id, "模型规划失败", error);
                    return;
                }
            }
        };

        if !task_is_active(&app, &task_id) {
            return;
        }

        if plan.done {
            let result = if plan.summary.trim().is_empty() {
                plan.observation
            } else {
                plan.summary
            };
            update_snapshot(&app.state::<RuntimeState>(), |s| {
                s.phase = "completed".into();
                s.message = result.clone();
            });
            emit(
                &app,
                event(
                    &task_id,
                    "completed",
                    "任务已验证完成",
                    result,
                    false,
                    true,
                    true,
                    None,
                ),
            );
            return;
        }

        let Some(step) = plan.step.as_ref() else {
            finish_with_error(&app, &task_id, "计划无效", "模型未返回可执行动作".into());
            return;
        };
        let foreground_now = foreground_window_title().unwrap_or_else(|| "Unknown".into());
        let foreground_handle_now = unsafe { GetForegroundWindow().0 };
        if !is_context_switch_action(&step.action)
            && foreground_handle != 0
            && foreground_handle_now != foreground_handle
        {
            let detail = format!(
                "前台窗口已从“{foreground_window}”变为“{foreground_now}”，本步未执行；正在重新观察。"
            );
            history.push(detail.clone());
            emit(
                &app,
                event(
                    &task_id,
                    "observing",
                    "检测到用户接管",
                    detail,
                    false,
                    false,
                    true,
                    None,
                ),
            );
            continue;
        }
        update_snapshot(&app.state::<RuntimeState>(), |s| {
            s.phase = "executing".into();
            s.model_ready = true;
            s.message = format!("第 {step_index} 步 · {}：{}", step.action, step.target);
        });
        emit(
            &app,
            event(
                &task_id,
                "executing",
                format!("第 {step_index} 步 · {}", action_label(&step.action)),
                format!(
                    "{}\n目标：{}\n预期：{}",
                    plan.observation, step.target, step.expected_change
                ),
                false,
                false,
                true,
                None,
            ),
        );

        match execute_plan(&plan, &frame) {
            Ok(message) => history.push(format!(
                "第 {step_index} 步：{message}；预期：{}",
                step.expected_change
            )),
            Err(error) => {
                finish_with_error(&app, &task_id, "执行失败", error);
                return;
            }
        }
        std::thread::sleep(Duration::from_millis(650));
    }

    finish_with_error(
        &app,
        &task_id,
        "任务未完成",
        format!("已执行 {MAX_STEPS} 步，但模型仍未验证目标完成"),
    );
}

fn task_is_active(app: &AppHandle, task_id: &str) -> bool {
    let snapshot = app
        .state::<RuntimeState>()
        .snapshot
        .lock()
        .expect("runtime snapshot poisoned")
        .clone();
    snapshot.task_id.as_deref() == Some(task_id) && snapshot.phase != "stopped"
}

fn finish_with_error(app: &AppHandle, task_id: &str, title: &str, error: String) {
    update_snapshot(&app.state::<RuntimeState>(), |s| {
        s.phase = "error".into();
        s.message = error.clone();
    });
    emit(
        app,
        event(task_id, "error", title, error, false, true, false, None),
    );
}

fn action_label(action: &str) -> &str {
    let action = action.to_ascii_lowercase();
    if action.contains("open_app") || action.contains("launch") {
        "启动应用"
    } else if action.contains("activate") || action.contains("focus") {
        "切换窗口"
    } else if action.contains("click") {
        "点击"
    } else if action.contains("input") || action.contains("type") {
        "输入文本"
    } else if action.contains("key") || action.contains("press") {
        "按键"
    } else if action.contains("wait") {
        "等待"
    } else {
        "执行动作"
    }
}

fn is_context_switch_action(action: &str) -> bool {
    let action = action.to_lowercase();
    action.contains("activate")
        || action.contains("focus")
        || action.contains("window")
        || action.contains("open_app")
        || action.contains("launch")
        || action.contains("激活")
        || action.contains("窗口")
        || action.contains("启动应用")
        || action.contains("打开应用")
}

fn capture_primary() -> Result<ScreenFrame, String> {
    let monitors = Monitor::all().map_err(|e| format!("枚举显示器失败：{e}"))?;
    let monitor = monitors
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .ok_or("没有检测到主显示器")?;
    let rgba = monitor
        .capture_image()
        .map_err(|e| format!("主屏截图失败：{e}"))?;
    let width = rgba.width();
    let height = rgba.height();
    let scaled = DynamicImage::ImageRgba8(rgba).resize(1280, 720, FilterType::Triangle);
    let model_width = scaled.width();
    let model_height = scaled.height();
    let mut png = Vec::new();
    scaled
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| format!("PNG 编码失败：{e}"))?;
    Ok(ScreenFrame {
        png_base64: BASE64.encode(png),
        width,
        height,
        model_width,
        model_height,
    })
}

fn infer_plan(
    app: &AppHandle,
    task_id: &str,
    goal: &str,
    history: &[String],
    visible_windows: &[String],
    foreground_window: &str,
    step_index: usize,
    frame: &ScreenFrame,
    endpoint: &str,
) -> Result<NativePlan, String> {
    let prompt = "You are a local Windows computer-use agent operating in a repeated observe-act-verify loop. Reply ONLY valid JSON: {\"observation\":string,\"done\":boolean,\"summary\":string,\"step\":{\"action\":string,\"target\":string,\"x\":integer|null,\"y\":integer|null,\"text\":string|null,\"risk\":string,\"expected_change\":string}|null}. Infer the target application from the user's intent and the dynamic visible-window inventory; never rely on a built-in app-name mapping. First verify whether the user's whole goal is visibly complete. Set done=true only when the latest screenshot provides evidence; then provide a concise summary and step=null. When done=false, step MUST be a non-null executable action. Supported actions: activate_window (target MUST copy a distinctive substring from one inventory title), open_app (target is a concise Windows Search query when the intended app has no visible window), click (x/y in the supplied image), input (text), key (text: Enter/Escape/Tab/Backspace/Space/arrows), and wait. Prefer activate_window for an existing target; use open_app only when no listed window matches. Never claim completion merely because a window was activated, an app was launched, or one intermediate action succeeded.";
    let history_text = if history.is_empty() {
        "No actions executed yet.".to_string()
    } else {
        history.join("\n")
    };
    let windows_text = if visible_windows.is_empty() {
        "No other visible windows were discovered.".to_string()
    } else {
        visible_windows
            .iter()
            .enumerate()
            .map(|(index, title)| format!("{}. {title}", index + 1))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let payload = json!({
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": format!("User goal: {goal}\nLoop: {step_index}/12\nCurrent foreground window: {foreground_window}\nExecuted actions:\n{history_text}\nVisible windows:\n{windows_text}\nScreenshot coordinates: {} x {}", frame.model_width, frame.model_height)},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", frame.png_base64)}}
            ]}
        ],
        "temperature": 0.2,
        "max_tokens": 1200,
        "stream": true
    });
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client
        .post(endpoint)
        .json(&payload)
        .send()
        .map_err(|e| format!("llama-server 不可用（{endpoint}）：{e}"))?
        .error_for_status()
        .map_err(|e| format!("llama-server 返回错误：{e}"))?;
    let mut content = String::new();
    for line in BufReader::new(response).lines() {
        let line = line.map_err(|e| format!("读取模型流失败：{e}"))?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        if let Ok(value) = serde_json::from_str::<Value>(data) {
            if let Some(delta) = value
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
            {
                content.push_str(delta);
                emit(
                    app,
                    event(
                        task_id,
                        "planning",
                        "模型输出中",
                        content.clone(),
                        false,
                        false,
                        true,
                        None,
                    ),
                );
            }
        }
    }
    if content.trim().is_empty() {
        return Err("模型流没有返回内容".into());
    }
    let json_text =
        extract_model_json(&content).ok_or_else(|| "模型响应中未找到 JSON 对象".to_string())?;
    let parsed: Value = serde_json::from_str(&json_text)
        .map_err(|error| format!("模型响应不是有效 JSON：{error}"))?;
    let observation = parsed
        .get("observation")
        .and_then(Value::as_str)
        .unwrap_or("模型已完成屏幕观察")
        .to_string();
    let done = parsed.get("done").and_then(Value::as_bool).unwrap_or(false);
    let summary = parsed
        .get("summary")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let step_value = parsed
        .get("step")
        .filter(|value| !value.is_null())
        .cloned()
        .or_else(|| {
            parsed
                .get("steps")
                .and_then(Value::as_array)
                .and_then(|steps| steps.first())
                .cloned()
        });
    let step = step_value
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| format!("模型步骤格式错误：{e}"))?;
    if !done && step.is_none() {
        return Err("模型既未确认任务完成，也没有返回下一步动作".into());
    }
    Ok(NativePlan {
        observation,
        summary,
        done,
        step,
    })
}

fn infer_window_intent(
    goal: &str,
    visible_windows: &[String],
    endpoint: &str,
) -> Result<Option<NativePlan>, String> {
    let inventory = if visible_windows.is_empty() {
        "No visible windows discovered.".into()
    } else {
        visible_windows
            .iter()
            .enumerate()
            .map(|(index, title)| format!("{}. {title}", index + 1))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let payload = json!({
        "messages": [
            {"role": "system", "content": "Resolve Windows application intent without a built-in alias list. Prefer a visible window that matches the user's task. If none matches but the user clearly names an app, provide a concise Windows Search query. Reply ONLY JSON: {\"target_window\":string|null,\"launch_query\":string|null,\"reason\":string}. target_window must copy a distinctive substring from exactly one supplied title. Never invent a window title."},
            {"role": "user", "content": format!("User goal: {goal}\nVisible windows:\n{inventory}")}
        ],
        "temperature": 0.0,
        "max_tokens": 240,
        "stream": false
    });
    let response: Value = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|error| error.to_string())?
        .post(endpoint)
        .json(&payload)
        .send()
        .map_err(|error| format!("窗口意图模型不可用：{error}"))?
        .error_for_status()
        .map_err(|error| format!("窗口意图模型返回错误：{error}"))?
        .json()
        .map_err(|error| format!("窗口意图响应解析失败：{error}"))?;
    let content = response
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or("窗口意图响应没有内容")?;
    let json_text = extract_model_json(content).ok_or("窗口意图响应中没有 JSON")?;
    let parsed: Value =
        serde_json::from_str(&json_text).map_err(|error| format!("窗口意图 JSON 无效：{error}"))?;
    let reason = parsed
        .get("reason")
        .and_then(Value::as_str)
        .unwrap_or("根据用户目标选择最匹配的可见窗口");
    if let Some(requested) = parsed.get("target_window").and_then(Value::as_str) {
        let requested = requested.trim().to_lowercase();
        if !requested.is_empty() {
            let Some(actual_title) = visible_windows.iter().find(|title| {
                let title = title.to_lowercase();
                title.contains(&requested) || requested.contains(&title)
            }) else {
                return Err(format!("模型选择了清单外的窗口：{requested}"));
            };
            return Ok(Some(intent_plan(
                reason,
                "activate_window",
                actual_title,
                format!("窗口“{actual_title}”成为当前活动窗口"),
            )));
        }
    }
    if let Some(query) = parsed.get("launch_query").and_then(Value::as_str) {
        let query = query.trim();
        if !query.is_empty() && query.chars().count() <= 120 {
            return Ok(Some(intent_plan(
                reason,
                "open_app",
                query,
                format!("启动应用“{query}”并出现可见窗口"),
            )));
        }
    }
    Ok(None)
}

fn intent_plan(reason: &str, action: &str, target: &str, expected_change: String) -> NativePlan {
    NativePlan {
        observation: reason.into(),
        summary: String::new(),
        done: false,
        step: Some(PlanStep {
            action: action.into(),
            target: target.into(),
            x: None,
            y: None,
            text: None,
            risk: String::new(),
            expected_change,
            requires_confirmation: false,
        }),
    }
}

/// Pull a JSON object out of model text that may be wrapped in markdown fences
/// or surrounded by short prose.
fn extract_model_json(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    let without_fence = if trimmed.starts_with("```") {
        let after_open = trimmed
            .trim_start_matches('`')
            .trim_start()
            .trim_start_matches("json")
            .trim_start_matches("JSON")
            .trim_start();
        after_open
            .strip_suffix("```")
            .map(|s| s.trim())
            .unwrap_or(after_open)
            .trim()
    } else {
        trimmed
    };
    let bytes = without_fence.as_bytes();
    let start = without_fence.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (idx, ch) in without_fence[start..].char_indices() {
        let i = start + idx;
        if in_string {
            if escape {
                escape = false;
            } else if ch == '\\' {
                escape = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    // Safe: start..i+ch.len_utf8() are char boundaries from char_indices.
                    let end = i + ch.len_utf8();
                    if end <= bytes.len() {
                        return Some(without_fence[start..end].to_string());
                    }
                }
            }
            _ => {}
        }
    }
    None
}

fn execute_plan(plan: &NativePlan, frame: &ScreenFrame) -> Result<String, String> {
    let settings = Settings::default();
    let mut enigo =
        Enigo::new(&settings).map_err(|error| format!("初始化输入控制失败：{error}"))?;
    let step = plan.step.as_ref().ok_or("计划没有可执行步骤")?;
    let action = step.action.to_lowercase();

    if action.contains("open_app")
        || action.contains("launch")
        || action.contains("启动应用")
        || action.contains("打开应用")
    {
        let query = step.target.trim();
        if query.is_empty() {
            return Err("启动应用动作缺少搜索词".into());
        }
        enigo
            .key(enigo::Key::Meta, Direction::Click)
            .map_err(|error| format!("打开 Windows 搜索失败：{error}"))?;
        std::thread::sleep(Duration::from_millis(350));
        enigo
            .text(query)
            .map_err(|error| format!("输入应用搜索词失败：{error}"))?;
        std::thread::sleep(Duration::from_millis(450));
        enigo
            .key(enigo::Key::Return, Direction::Click)
            .map_err(|error| format!("启动应用失败：{error}"))?;
        std::thread::sleep(Duration::from_millis(900));
    } else if action.contains("activate")
        || action.contains("focus")
        || action.contains("window")
        || action.contains("激活")
        || action.contains("窗口")
    {
        activate_window(&step.target)?;
    } else if action.contains("click") || action.contains("点击") {
        let model_x = step.x.ok_or("点击动作缺少 x 坐标")?;
        let model_y = step.y.ok_or("点击动作缺少 y 坐标")?;
        let x = ((model_x as f64) * frame.width as f64 / frame.model_width as f64).round() as i32;
        let y = ((model_y as f64) * frame.height as f64 / frame.model_height as f64).round() as i32;
        enigo
            .move_mouse(x, y, Coordinate::Abs)
            .map_err(|error| format!("移动鼠标失败：{error}"))?;
        enigo
            .button(Button::Left, Direction::Click)
            .map_err(|error| format!("点击失败：{error}"))?;
    } else if action.contains("type")
        || action.contains("input")
        || action.contains("输入")
        || action.contains("填写")
    {
        let text = step.text.as_deref().ok_or("输入动作缺少文本")?;
        enigo
            .text(text)
            .map_err(|error| format!("输入文本失败：{error}"))?;
    } else if action.contains("key") || action.contains("press") || action.contains("按") {
        let key = step
            .text
            .as_deref()
            .or(Some(step.target.as_str()))
            .unwrap_or_default()
            .to_lowercase();
        let key = match key.as_str() {
            "enter" | "return" | "回车" | "确认" => enigo::Key::Return,
            "escape" | "esc" | "退出" => enigo::Key::Escape,
            "tab" | "制表" => enigo::Key::Tab,
            "backspace" | "退格" => enigo::Key::Backspace,
            "space" | "空格" => enigo::Key::Space,
            "up" | "上" => enigo::Key::UpArrow,
            "down" | "下" => enigo::Key::DownArrow,
            "left" | "左" => enigo::Key::LeftArrow,
            "right" | "右" => enigo::Key::RightArrow,
            value => value
                .chars()
                .next()
                .map(enigo::Key::Unicode)
                .ok_or("按键动作缺少按键")?,
        };
        enigo
            .key(key, Direction::Click)
            .map_err(|error| format!("按键失败：{error}"))?;
    } else if action == "wait" || action.contains("等待") || action.contains("观察") {
        std::thread::sleep(Duration::from_millis(250));
    } else {
        return Err(format!("暂不支持的动作：{}", step.action));
    }

    Ok(format!("{} · {}", action_label(&step.action), step.target))
}

struct WindowSearch {
    query: String,
    found: HWND,
}

struct WindowInventory {
    titles: Vec<String>,
}

unsafe fn visible_window_title(hwnd: HWND) -> Option<String> {
    if !IsWindowVisible(hwnd).as_bool() {
        return None;
    }
    let mut title = vec![0u16; 512];
    let length = GetWindowTextW(hwnd, &mut title);
    (length > 0)
        .then(|| {
            String::from_utf16_lossy(&title[..length as usize])
                .trim()
                .to_string()
        })
        .filter(|title| !title.is_empty())
}

unsafe extern "system" fn find_window_callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let search = &mut *(lparam.0 as *mut WindowSearch);
    if let Some(value) = visible_window_title(hwnd) {
        if value.to_lowercase().contains(&search.query.to_lowercase()) {
            search.found = hwnd;
            return BOOL(0);
        }
    }
    BOOL(1)
}

unsafe extern "system" fn collect_windows_callback(hwnd: HWND, lparam: LPARAM) -> BOOL {
    let inventory = &mut *(lparam.0 as *mut WindowInventory);
    if let Some(title) = visible_window_title(hwnd) {
        inventory.titles.push(title);
    }
    BOOL(1)
}

fn visible_window_titles() -> Result<Vec<String>, String> {
    let mut inventory = WindowInventory { titles: Vec::new() };
    unsafe {
        EnumWindows(
            Some(collect_windows_callback),
            LPARAM(&mut inventory as *mut _ as isize),
        )
        .map_err(|error| format!("枚举可见窗口失败：{error}"))?;
    }
    inventory.titles.sort_by_key(|title| title.to_lowercase());
    inventory
        .titles
        .dedup_by(|left, right| left.eq_ignore_ascii_case(right));
    Ok(inventory.titles)
}

fn foreground_window_title() -> Option<String> {
    unsafe { visible_window_title(GetForegroundWindow()) }
}

fn activate_window(target: &str) -> Result<(), String> {
    let query = target.trim();
    if query.is_empty() {
        return Err("激活窗口动作缺少窗口标题".into());
    }
    let mut search = WindowSearch {
        query: query.into(),
        found: HWND(0),
    };
    unsafe {
        EnumWindows(
            Some(find_window_callback),
            LPARAM(&mut search as *mut _ as isize),
        )
        .map_err(|error| format!("枚举窗口失败：{error}"))?;
        if search.found.0 == 0 {
            return Err(format!("没有找到窗口：{query}"));
        }
        let _ = ShowWindow(search.found, SW_RESTORE);
        if !SetForegroundWindow(search.found).as_bool() {
            return Err(format!("无法激活窗口：{query}"));
        }
    }
    Ok(())
}

fn now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    format!("{secs}Z")
}

pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeState::default())
        .manage(ModelState::default())
        .setup(|app| {
            let handle = app.handle().clone();
            start_emergency_stop_monitor(handle.clone());
            std::thread::spawn(move || start_model(handle));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_runtime,
            get_model_config,
            set_model_config,
            run_task,
            pause_runtime,
            stop_runtime
        ])
        .run(tauri::generate_context!())
        .expect("error while running baodou desktop");
}
