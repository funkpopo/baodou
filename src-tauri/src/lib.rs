use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use enigo::{Button, Coordinate, Direction, Enigo, Keyboard, Mouse, Settings};
use image::{imageops::FilterType, DynamicImage, ImageFormat};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    env,
    io::{BufRead, BufReader, Cursor},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::Duration,
};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;
use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_CONTROL, VK_ESCAPE, VK_MENU,
};
use windows::Win32::UI::WindowsAndMessaging::{
    BringWindowToTop, EnumWindows, GetForegroundWindow, GetWindowTextW, GetWindowThreadProcessId,
    IsWindowVisible, SetForegroundWindow, ShowWindow, SW_RESTORE,
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
        if step_index == 1 {
            emit(
                &app,
                event(
                    &task_id,
                    "planning",
                    "识别任务目标",
                    format!(
                        "正在根据任务目标绑定应用窗口；当前前台：{}，可见窗口：{} 个",
                        foreground_window,
                        visible_windows.len()
                    ),
                    false,
                    false,
                    true,
                    None,
                ),
            );
        }
        let explicit_launch = if step_index == 1 && has_explicit_launch_intent(&goal) {
            launch_plan_from_goal(&goal, "用户明确要求启动应用")
        } else {
            None
        };
        let plan = if let Some(plan) = explicit_launch {
            plan
        } else if step_index == 1 {
            // Resolve the task's application/window target before UI planning.
            // This is a harness boundary: an arbitrary visible system surface
            // cannot become a target merely because it appears in the frame.
            match infer_window_intent(&goal, &visible_windows, &endpoint) {
                Ok(Some(plan)) => plan,
                Ok(None) => {
                    if let Some(plan) = launch_plan_from_goal(&goal, "未解析到可操作的现有窗口")
                    {
                        plan
                    } else {
                        finish_with_error(
                            &app,
                            &task_id,
                            "任务目标解析失败",
                            "无法从任务中解析应用启动目标".into(),
                        );
                        return;
                    }
                }
                Err(_) => match infer_plan(
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
                        finish_with_error(&app, &task_id, "模型规划失败", error);
                        return;
                    }
                },
            }
        } else {
            match infer_plan(
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
                    let should_resolve_intent = (step_index == 1 && error.contains("下一步动作"))
                        || error.contains("重复激活当前前台窗口");
                    if should_resolve_intent {
                        match infer_window_intent(&goal, &visible_windows, &endpoint) {
                            Ok(Some(plan)) => plan,
                            Ok(None) => {
                                if let Some(plan) =
                                    launch_plan_from_goal(&goal, "模型未返回窗口意图")
                                {
                                    plan
                                } else {
                                    finish_with_error(&app, &task_id, "模型规划失败", error);
                                    return;
                                }
                            }
                            Err(intent_error) => {
                                if let Some(plan) = launch_plan_from_goal(&goal, &intent_error) {
                                    plan
                                } else {
                                    finish_with_error(
                                        &app,
                                        &task_id,
                                        "窗口意图判断失败",
                                        format!("{error}；{intent_error}"),
                                    );
                                    return;
                                }
                            }
                        }
                    } else {
                        finish_with_error(&app, &task_id, "模型规划失败", error);
                        return;
                    }
                }
            }
        };

        if step_index == 1 {
            let binding_detail = plan
                .step
                .as_ref()
                .map(|step| {
                    format!(
                        "已绑定目标：{} · {}",
                        action_label(&step.action),
                        step.target
                    )
                })
                .unwrap_or_else(|| "目标获取完成，正在进入验证".into());
            emit(
                &app,
                event(
                    &task_id,
                    "planning",
                    "目标已确定",
                    binding_detail,
                    false,
                    false,
                    true,
                    None,
                ),
            );
        }

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
    } else if action.contains("clear") || action.contains("清空") || action.contains("清除") {
        "清空字段"
    } else if action.contains("replace") || action.contains("替换") {
        "替换文本"
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
    is_window_activation_action(&action)
        || action.contains("open_app")
        || action.contains("launch")
        || action.contains("启动应用")
        || action.contains("打开应用")
}

fn is_window_activation_action(action: &str) -> bool {
    let action = action.to_lowercase();
    action.contains("activate")
        || action.contains("focus")
        || action.contains("window")
        || action.contains("激活")
        || action.contains("窗口")
}

fn normalize_window_action(plan: &mut NativePlan, visible_windows: &[String]) {
    let Some(step) = plan.step.as_mut() else {
        return;
    };
    if !is_window_activation_action(&step.action) || step.target.trim().is_empty() {
        return;
    }
    let requested = step.target.trim().to_lowercase();
    if let Some(actual_title) = visible_windows.iter().find(|title| {
        let title = title.to_lowercase();
        title.contains(&requested) || requested.contains(&title)
    }) {
        step.target = actual_title.clone();
        return;
    }
    let query = step.target.trim().to_string();
    step.action = "open_app".into();
    step.x = None;
    step.y = None;
    step.text = None;
    step.expected_change = format!("通过 Windows Search 启动“{query}”并出现可见窗口");
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
    let prompt = "You are a local Windows computer-use agent in an observe-act-verify loop. Return exactly ONE next action, never a sequence or a full future workflow. Do not output JSON. Put every field on its own line in this exact order:\nSTATUS: CONTINUE or DONE\nOBSERVATION: only what the screenshot currently proves\nACTION: activate_window, open_app, click, input, replace, clear_search_bar, key, wait, or none\nTARGET: window title, app search query, or UI target\nX: integer or blank\nY: integer or blank\nTEXT: input text/key or blank\nEXPECTED: expected visible change after this one action\nSUMMARY: completion summary or blank\nEND_PLAN\nWrite STATUS exactly once and finish at END_PLAN. Do not predict later actions and do not repeat the template. Infer the application from user intent and the live window inventory, never from a built-in mapping. The window inventory is observation data, not an allow-list. Desktop, shell, settings, notifications, input-method windows, and text-input hosts must not be selected without positive evidence that they are the task application. For activate_window, copy a distinctive substring from a real inventory title and never activate the already-current foreground window. Use open_app when no listed window is clearly relevant. For a search or form field, first use clear_search_bar when stale text may exist; use replace to select all and enter new text atomically. Plain input appends and must only be used when appending is intended. STATUS DONE is valid only when the latest screenshot proves the whole goal is complete; then ACTION must be none. Otherwise STATUS must be CONTINUE and ACTION must be executable. Never mark done merely because a window opened or an intermediate action succeeded.";
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
        "temperature": 0.0,
        "max_tokens": 512,
        "stop": ["END_PLAN"],
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
                let (bounded, terminated) = first_model_plan(&content);
                emit(
                    app,
                    event(
                        task_id,
                        "planning",
                        "模型输出中",
                        bounded.clone(),
                        false,
                        false,
                        true,
                        None,
                    ),
                );
                if terminated {
                    content = bounded;
                    break;
                }
            }
        }
    }
    if content.trim().is_empty() {
        return Err("模型流没有返回内容".into());
    }
    let mut plan = parse_model_plan(&first_model_plan(&content).0)?;
    normalize_window_action(&mut plan, visible_windows);
    if let Some(step) = plan.step.as_ref() {
        let target = step.target.trim().to_lowercase();
        let foreground = foreground_window.trim().to_lowercase();
        if is_window_activation_action(&step.action)
            && !target.is_empty()
            && !foreground.is_empty()
            && (foreground.contains(&target) || target.contains(&foreground))
        {
            return Err(format!(
                "模型试图重复激活当前前台窗口“{foreground_window}”，未产生可验证的状态变化"
            ));
        }
    }
    Ok(plan)
}

fn parse_model_plan(content: &str) -> Result<NativePlan, String> {
    let fields = tagged_fields(&first_model_plan(content).0);
    let status = tagged_value(&fields, &["STATUS", "状态"]).to_uppercase();
    let action = tagged_value(&fields, &["ACTION", "动作"]);
    let done = ["DONE", "COMPLETE", "COMPLETED", "完成"]
        .iter()
        .any(|value| status.contains(value))
        || ["done", "complete", "completed", "完成"]
            .iter()
            .any(|value| action.eq_ignore_ascii_case(value));
    let observation = tagged_value(&fields, &["OBSERVATION", "观察"]);
    let summary = tagged_value(&fields, &["SUMMARY", "总结", "结果"]);

    if done {
        return Ok(NativePlan {
            observation: if observation.is_empty() {
                "模型确认目标已经完成".into()
            } else {
                observation
            },
            summary,
            done: true,
            step: None,
        });
    }
    if action.is_empty() || action.eq_ignore_ascii_case("none") || action == "无" {
        return Err("模型既未确认任务完成，也没有返回下一步动作".into());
    }
    let target = tagged_value(&fields, &["TARGET", "目标"]);
    let text = tagged_value(&fields, &["TEXT", "文本", "按键"]);
    let expected_change = tagged_value(&fields, &["EXPECTED", "预期"]);
    let x = tagged_value(&fields, &["X"]);
    let y = tagged_value(&fields, &["Y"]);
    Ok(NativePlan {
        observation: if observation.is_empty() {
            "模型已选择下一步动作".into()
        } else {
            observation
        },
        summary,
        done: false,
        step: Some(PlanStep {
            action,
            target,
            x: x.parse().ok(),
            y: y.parse().ok(),
            text: (!text.is_empty()).then_some(text),
            risk: String::new(),
            expected_change,
            requires_confirmation: false,
        }),
    })
}

fn tagged_fields(content: &str) -> HashMap<String, String> {
    let tail = content
        .rsplit_once("</think>")
        .map(|(_, tail)| tail)
        .unwrap_or(content);
    let normalized = tail.replace(['；', ';'], "\n");
    normalized
        .lines()
        .filter_map(|line| {
            let line = line
                .trim()
                .trim_matches('`')
                .trim_start_matches(['-', '*', ' ']);
            let pair = line.split_once(':').or_else(|| line.split_once('：'))?;
            let key = pair.0.trim().to_uppercase();
            (!key.is_empty()).then(|| (key, pair.1.trim().to_string()))
        })
        .fold(HashMap::new(), |mut fields, (key, value)| {
            fields.entry(key).or_insert(value);
            fields
        })
}

/// Keep one model turn to one action even when a small model starts repeating
/// the protocol template. The server normally stops before emitting END_PLAN;
/// the repeated STATUS boundary is the local fallback for non-compliant models.
fn first_model_plan(content: &str) -> (String, bool) {
    let tail_start = content
        .rfind("</think>")
        .map(|index| index + "</think>".len())
        .unwrap_or(0);
    let tail = &content[tail_start..];
    let marker_boundary = tail.find("END_PLAN").map(|index| tail_start + index);
    let mut statuses = Vec::new();
    for tag in ["STATUS:", "STATUS：", "状态:", "状态："] {
        for (index, _) in tail.match_indices(tag) {
            let prefix = &tail[..index];
            let at_field_start = prefix
                .trim_end_matches(char::is_whitespace)
                .chars()
                .next_back()
                .map(|ch| matches!(ch, '\n' | '\r' | ';' | '；'))
                .unwrap_or(true);
            if at_field_start {
                statuses.push(tail_start + index);
            }
        }
    }
    statuses.sort_unstable();
    statuses.dedup();
    let repeated_boundary = statuses.get(1).copied();
    let boundary = match (marker_boundary, repeated_boundary) {
        (Some(marker), Some(repeated)) => Some(marker.min(repeated)),
        (Some(marker), None) => Some(marker),
        (None, Some(repeated)) => Some(repeated),
        (None, None) => None,
    };
    match boundary {
        Some(index) => (content[..index].trim().to_string(), true),
        None => (content.trim().to_string(), false),
    }
}

fn tagged_value(fields: &HashMap<String, String>, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| fields.get(&key.to_uppercase()))
        .cloned()
        .unwrap_or_default()
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
            {"role": "system", "content": "Resolve the user's application target as a harness step, without any built-in application or window blacklist. Do not output JSON. Reply with exactly three tagged lines: DECISION: WINDOW, LAUNCH, or NONE; TARGET: a real visible window-title substring for WINDOW, a concise Windows Search query for LAUNCH, or blank; REASON: one short reason. Choose WINDOW only when the title itself provides positive evidence that it is the application needed for the user's goal. A visible title is not evidence merely because it exists, and a desktop, shell, settings, notification, input, or utility surface must not be selected as a fallback. If no candidate is clearly relevant, choose LAUNCH. Never invent a window title."},
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
    let content = match response.pointer("/choices/0/message/content") {
        Some(Value::String(content)) if !content.trim().is_empty() => content.clone(),
        Some(content) if !content.is_null() => content.to_string(),
        _ => response
            .pointer("/choices/0/message/reasoning_content")
            .or_else(|| response.pointer("/choices/0/text"))
            .and_then(Value::as_str)
            .filter(|content| !content.trim().is_empty())
            .map(str::to_string)
            .ok_or("窗口意图响应没有内容")?,
    };
    let fields = tagged_fields(&content);
    let decision = tagged_value(&fields, &["DECISION", "决定", "类型"]);
    let target = tagged_value(&fields, &["TARGET", "目标", "窗口", "应用"]);
    let reason = tagged_value(&fields, &["REASON", "原因"]);
    if !decision.is_empty() {
        return resolve_window_intent_fields(&decision, &target, &reason, visible_windows);
    }
    resolve_plain_window_intent(&content, visible_windows)
}

fn resolve_window_intent_fields(
    decision: &str,
    target: &str,
    reason: &str,
    visible_windows: &[String],
) -> Result<Option<NativePlan>, String> {
    let decision = decision.trim().to_uppercase();
    let target = target.trim();
    let reason = if reason.trim().is_empty() {
        "根据用户目标判断目标应用"
    } else {
        reason.trim()
    };
    if decision.contains("NONE") || decision.contains("无") {
        return Ok(None);
    }
    if decision.contains("WINDOW") || decision.contains("窗口") {
        let requested = target.to_lowercase();
        let Some(actual_title) = visible_windows.iter().find(|title| {
            let title = title.to_lowercase();
            title.contains(&requested) || requested.contains(&title)
        }) else {
            return Err(format!("模型选择了清单外的窗口：{target}"));
        };
        return Ok(Some(intent_plan(
            reason,
            "activate_window",
            actual_title,
            format!("窗口“{actual_title}”成为当前活动窗口"),
        )));
    }
    if decision.contains("LAUNCH") || decision.contains("启动") || decision.contains("打开") {
        if target.is_empty() || target.chars().count() > 120 {
            return Err("模型没有返回有效的应用启动查询".into());
        }
        return Ok(Some(intent_plan(
            reason,
            "open_app",
            target,
            format!("启动应用“{target}”并出现可见窗口"),
        )));
    }
    resolve_plain_window_intent(target, visible_windows)
}

fn resolve_plain_window_intent(
    content: &str,
    visible_windows: &[String],
) -> Result<Option<NativePlan>, String> {
    let tail = content
        .rsplit_once("</think>")
        .map(|(_, tail)| tail)
        .unwrap_or(content)
        .trim();
    let candidate = tail
        .lines()
        .rev()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or("")
        .trim_matches(|ch| matches!(ch, '`' | '"' | '\'' | '“' | '”'));
    let candidate = ["target:", "app:", "窗口：", "应用："]
        .iter()
        .find_map(|prefix| candidate.strip_prefix(prefix))
        .unwrap_or(candidate)
        .trim();
    if candidate.is_empty() || candidate.chars().count() > 80 {
        return Err("窗口意图响应中没有可用标签或简短目标".into());
    }
    if let Some(actual_title) = visible_windows.iter().find(|title| {
        let title = title.to_lowercase();
        let candidate = candidate.to_lowercase();
        title.contains(&candidate) || candidate.contains(&title)
    }) {
        return Ok(Some(intent_plan(
            "从模型纯文本响应中识别到当前可见窗口",
            "activate_window",
            actual_title,
            format!("窗口“{actual_title}”成为当前活动窗口"),
        )));
    }
    if candidate.contains(['。', '！', '？', '!', '?', ';', '；']) {
        return Err("窗口意图响应不是结构化结果，也不是简短应用名称".into());
    }
    Ok(Some(intent_plan(
        "模型返回了简短应用意图，使用 Windows Search 启动",
        "open_app",
        candidate,
        format!("启动应用“{candidate}”并出现可见窗口"),
    )))
}

fn launch_plan_from_goal(goal: &str, reason: &str) -> Option<NativePlan> {
    let query = extract_launch_query(goal)?;
    Some(intent_plan(
        &format!("窗口意图模型格式无效（{reason}），从用户指令中提取目标应用。"),
        "open_app",
        &query,
        format!("启动应用“{query}”并出现可见窗口"),
    ))
}

fn has_explicit_launch_intent(goal: &str) -> bool {
    let trimmed = goal.trim_start();
    ["打开", "启动", "运行"]
        .iter()
        .any(|prefix| trimmed.starts_with(prefix))
}

fn extract_launch_query(goal: &str) -> Option<String> {
    let goal = goal.trim();
    let prefixes = [
        "打开",
        "启动",
        "运行",
        "切换到",
        "切换至",
        "使用",
        "用",
        "在",
    ];
    for prefix in prefixes {
        let Some(start) = goal.find(prefix) else {
            continue;
        };
        let rest = goal[start + prefix.len()..].trim_start_matches([' ', '@', '“', '"']);
        let terminators: &[&str] = if prefix == "在" {
            &["中", "里", "内"]
        } else {
            &[
                "并", "然后", "，", ",", "。", "搜索", "查找", "查看", "中", "里",
            ]
        };
        let end = terminators
            .iter()
            .filter_map(|marker| rest.find(marker))
            .min()
            .unwrap_or(rest.len());
        let query = rest[..end]
            .trim()
            .trim_matches([' ', '@', '“', '”', '"', '\'', ':', '：']);
        if !query.is_empty() && query.chars().count() <= 60 {
            return Some(query.into());
        }
    }
    None
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
        launch_app_with_search(&mut enigo, &step.target)?;
    } else if action.contains("activate")
        || action.contains("focus")
        || action.contains("window")
        || action.contains("激活")
        || action.contains("窗口")
    {
        if let Err(error) = activate_window(&step.target) {
            if error.starts_with("没有找到窗口：") {
                launch_app_with_search(&mut enigo, &step.target)?;
                return Ok(format!(
                    "目标窗口不存在，已通过 Windows Search 启动 · {}",
                    step.target
                ));
            }
            return Err(error);
        }
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
    } else if action.contains("clear_search_bar")
        || action.contains("clear_search")
        || action.contains("清空搜索")
        || action.contains("清除搜索")
    {
        clear_focused_text(&mut enigo)?;
    } else if action.contains("replace") || action.contains("替换") {
        clear_focused_text(&mut enigo)?;
        let text = step.text.as_deref().ok_or("替换动作缺少文本")?;
        enigo
            .text(text)
            .map_err(|error| format!("替换文本失败：{error}"))?;
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

fn clear_focused_text(enigo: &mut Enigo) -> Result<(), String> {
    enigo
        .key(enigo::Key::Control, Direction::Press)
        .map_err(|error| format!("选择文本失败：{error}"))?;
    enigo
        .key(enigo::Key::Unicode('a'), Direction::Click)
        .map_err(|error| format!("选择文本失败：{error}"))?;
    enigo
        .key(enigo::Key::Control, Direction::Release)
        .map_err(|error| format!("选择文本失败：{error}"))?;
    enigo
        .key(enigo::Key::Backspace, Direction::Click)
        .map_err(|error| format!("清空文本失败：{error}"))
}

fn launch_app_with_search(enigo: &mut Enigo, query: &str) -> Result<(), String> {
    let query = query.trim();
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
    Ok(())
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
        let current_thread = GetCurrentThreadId();
        let target_thread = GetWindowThreadProcessId(search.found, None);
        let attached = target_thread != 0
            && target_thread != current_thread
            && AttachThreadInput(current_thread, target_thread, true).as_bool();
        let requested = SetForegroundWindow(search.found).as_bool();
        let _ = BringWindowToTop(search.found);
        if attached {
            let _ = AttachThreadInput(current_thread, target_thread, false);
        }
        for _ in 0..12 {
            if GetForegroundWindow() == search.found {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        if !requested {
            return Err(format!("无法激活窗口：{query}（Windows 拒绝切换前台）"));
        }
        Err(format!("窗口激活调用成功，但“{query}”未成为前台窗口"))
    }
}

fn now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default();
    format!("{secs}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_generic_chinese_launch_intent() {
        assert_eq!(
            extract_launch_query("打开浏览器搜索今天的天气").as_deref(),
            Some("浏览器")
        );
        assert_eq!(
            extract_launch_query("在 VS Code 中提交当前代码").as_deref(),
            Some("VS Code")
        );
        assert_eq!(
            extract_launch_query("使用 Chrome 查看项目页面").as_deref(),
            Some("Chrome")
        );
        assert!(has_explicit_launch_intent("打开浏览器搜索今天的天气"));
        assert!(has_explicit_launch_intent("启动记事本"));
        assert!(!has_explicit_launch_intent("在浏览器中搜索今天的天气"));
    }

    #[test]
    fn plain_intent_prefers_a_real_visible_window() {
        let windows = vec!["weather — Browser".to_string(), "baodou".to_string()];
        let plan = resolve_plain_window_intent("weather — Browser", &windows)
            .expect("plain intent should parse")
            .expect("a plan should be produced");
        let step = plan.step.expect("plan should contain a step");
        assert_eq!(step.action, "activate_window");
        assert_eq!(step.target, "weather — Browser");
    }

    #[test]
    fn plain_app_name_becomes_dynamic_search_query() {
        let plan = resolve_plain_window_intent("浏览器", &[])
            .expect("plain app intent should parse")
            .expect("a plan should be produced");
        let step = plan.step.expect("plan should contain a step");
        assert_eq!(step.action, "open_app");
        assert_eq!(step.target, "浏览器");
    }

    #[test]
    fn parses_tagged_computer_use_action_without_json() {
        let plan = parse_model_plan(
            "STATUS: CONTINUE\nOBSERVATION: 浏览器尚未打开\nACTION: open_app\nTARGET: 浏览器\nX:\nY:\nTEXT:\nEXPECTED: 浏览器窗口出现\nSUMMARY:",
        )
        .expect("tagged action should parse");
        assert!(!plan.done);
        let step = plan.step.expect("action should contain a step");
        assert_eq!(step.action, "open_app");
        assert_eq!(step.target, "浏览器");
    }

    #[test]
    fn parses_idempotent_field_actions() {
        let plan = parse_model_plan(
            "STATUS: CONTINUE\nOBSERVATION: 搜索框中有旧文本\nACTION: clear_search_bar\nTARGET: search bar\nEXPECTED: 搜索框为空",
        )
        .expect("clear action should parse");
        assert_eq!(plan.step.expect("step").action, "clear_search_bar");

        let plan = parse_model_plan(
            "STATUS: CONTINUE\nOBSERVATION: 搜索框已清空\nACTION: replace\nTARGET: search bar\nTEXT: 今天的天气\nEXPECTED: 搜索框包含准确查询词",
        )
        .expect("replace action should parse");
        assert_eq!(plan.step.expect("step").action, "replace");
    }

    #[test]
    fn parses_tagged_completion_without_json() {
        let plan = parse_model_plan(
            "STATUS: DONE\nOBSERVATION: 天气结果已显示\nACTION: none\nSUMMARY: 已打开浏览器并搜索今天的天气",
        )
        .expect("tagged completion should parse");
        assert!(plan.done);
        assert!(plan.step.is_none());
        assert_eq!(plan.summary, "已打开浏览器并搜索今天的天气");
    }

    #[test]
    fn repeated_model_protocol_uses_only_the_first_action() {
        let response = "STATUS: CONTINUE; OBSERVATION: 浏览器尚未打开; ACTION: open_app; TARGET: Microsoft Edge; X:; Y:; TEXT:; EXPECTED: 浏览器打开; SUMMARY:; STATUS: DONE; OBSERVATION: 猜测天气已显示; ACTION: none; SUMMARY: 已完成";
        let (bounded, terminated) = first_model_plan(response);
        assert!(terminated);
        assert_eq!(bounded.matches("STATUS:").count(), 1);

        let plan = parse_model_plan(response).expect("first tagged action should parse");
        assert!(!plan.done);
        let step = plan.step.expect("first action should be preserved");
        assert_eq!(step.action, "open_app");
        assert_eq!(step.target, "Microsoft Edge");
    }

    #[test]
    fn end_plan_marker_terminates_streamed_plan() {
        let response = "STATUS: DONE\nOBSERVATION: 天气结果已显示\nACTION: none\nSUMMARY: 已完成\nEND_PLAN\nSTATUS: CONTINUE";
        let (bounded, terminated) = first_model_plan(response);
        assert!(terminated);
        assert!(!bounded.contains("END_PLAN"));
        assert_eq!(bounded.matches("STATUS:").count(), 1);
    }

    #[test]
    fn missing_window_activation_becomes_dynamic_app_launch() {
        let mut plan = intent_plan(
            "目标浏览器尚未处于前台",
            "activate_window",
            "Microsoft Edge",
            "浏览器窗口成为前台窗口".into(),
        );
        normalize_window_action(&mut plan, &["baodou · 电脑操作助手".into()]);
        let step = plan.step.expect("plan should remain executable");
        assert_eq!(step.action, "open_app");
        assert_eq!(step.target, "Microsoft Edge");
    }

    #[test]
    fn visible_window_activation_uses_current_inventory_title() {
        let mut plan = intent_plan(
            "浏览器已经打开",
            "activate_window",
            "Edge",
            "浏览器窗口成为前台窗口".into(),
        );
        normalize_window_action(
            &mut plan,
            &["天气 - Microsoft Edge".into(), "baodou".into()],
        );
        let step = plan.step.expect("plan should remain executable");
        assert_eq!(step.action, "activate_window");
        assert_eq!(step.target, "天气 - Microsoft Edge");
    }
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
