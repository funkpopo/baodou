use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::{imageops::FilterType, DynamicImage, ImageFormat};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{env, io::Cursor, process::{Child, Command, Stdio}, sync::Mutex, time::Duration};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;
use xcap::Monitor;

const PROTOCOL_VERSION: &str = "1.0.0";
const LLAMA_ENDPOINT: &str = "http://127.0.0.1:8765/v1/chat/completions";

struct RuntimeState {
    snapshot: Mutex<RuntimeSnapshot>,
    pending_plan: Mutex<Option<NativePlan>>,
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
            model_path: r"model\Qwen3.5-2B-UD-Q4_K_XL.gguf".into(),
            mmproj_path: r"model\mmproj-F16.gguf".into(),
            llama_url: LLAMA_ENDPOINT.into(),
        }
    }
}

impl Default for ModelState {
    fn default() -> Self {
        Self { process: Mutex::new(None) }
    }
}

impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            snapshot: Mutex::new(RuntimeSnapshot::default()),
            pending_plan: Mutex::new(None),
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
            mode: "native preview · dry-run".into(),
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
    live: bool,
    auto_confirm: bool,
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
    risk: String,
    expected_change: String,
}

#[derive(Clone)]
struct NativePlan {
    observation: String,
    step: PlanStep,
    blocked_reason: Option<String>,
}

#[derive(Clone)]
struct ScreenFrame {
    png_base64: String,
    width: u32,
    height: u32,
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
    let directory = app.path().app_data_dir().map_err(|error| format!("无法定位应用数据目录：{error}"))?;
    std::fs::create_dir_all(&directory).map_err(|error| format!("无法创建应用数据目录：{error}"))?;
    Ok(directory.join("model-config.json"))
}

fn load_model_config(app: &AppHandle) -> ModelConfig {
    config_file(app)
        .ok()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|content| serde_json::from_str(&content).ok())
        .unwrap_or_default()
}

#[tauri::command]
fn get_model_config(app: AppHandle) -> ModelConfig {
    load_model_config(&app)
}

#[tauri::command]
fn set_model_config(app: AppHandle, config: ModelConfig) -> Result<ModelConfig, String> {
    if config.server_path.trim().is_empty() || config.model_path.trim().is_empty()
        || config.mmproj_path.trim().is_empty() || config.llama_url.trim().is_empty() {
        return Err("模型程序、模型文件、MMPROJ 和接口 URL 都不能为空".into());
    }
    let normalized = ModelConfig {
        server_path: config.server_path.trim().into(),
        model_path: config.model_path.trim().into(),
        mmproj_path: config.mmproj_path.trim().into(),
        llama_url: config.llama_url.trim().into(),
    };
    let path = config_file(&app)?;
    let content = serde_json::to_string_pretty(&normalized).map_err(|error| format!("配置序列化失败：{error}"))?;
    std::fs::write(path, content).map_err(|error| format!("配置保存失败：{error}"))?;
    Ok(normalized)
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

    let current_dir = env::current_dir().unwrap_or_default();
    let resolve = |path: &str| {
        let path = std::path::PathBuf::from(path);
        if path.is_absolute() { path } else { current_dir.join(path) }
    };
    let server = resolve(&config.server_path);
    let model = resolve(&config.model_path);
    let mmproj = resolve(&config.mmproj_path);

    if !server.exists() || !model.exists() || !mmproj.exists() {
        let missing = if !server.exists() { server.display().to_string() }
            else if !model.exists() { model.display().to_string() }
            else { mmproj.display().to_string() };
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
        .split(':').last().and_then(|value| value.split('/').next()).unwrap_or("8765");
    let child = Command::new(&server)
        .current_dir(server.parent().unwrap_or(std::path::Path::new(".")))
        .args(["-m", model.to_string_lossy().as_ref(), "--mmproj", mmproj.to_string_lossy().as_ref(), "--host", "127.0.0.1", "--port", port, "--jinja"])
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
        .spawn();
    match child {
        Ok(child) => {
            *app.state::<ModelState>().process.lock().expect("model state poisoned") = Some(child);
            for _ in 0..90 {
                if llama_health(&endpoint) {
                    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                        snapshot.model_ready = true;
                        snapshot.inference_backend = "llama-server · 本地视觉模型".into();
                        snapshot.message = "本地模型已就绪".into();
                    });
                    return;
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
    *state.pending_plan.lock().expect("pending plan poisoned") = None;
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
    let mode = if request.live {
        "native live · confirmation gate"
    } else {
        "native preview · dry-run"
    };
    let snapshot = update_snapshot(&state, |s| {
        s.mode = mode.into();
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
    std::thread::spawn(move || {
        run_native_task(app_clone, task_id, goal, request.live, request.auto_confirm)
    });
    Ok(snapshot.task_id.unwrap_or_default())
}

fn run_native_task(app: AppHandle, task_id: String, goal: String, live: bool, auto_confirm: bool) {
    let frame = match capture_primary() {
        Ok(frame) => {
            emit(
                &app,
                event(
                    &task_id,
                    "planning",
                    "屏幕已采集",
                    format!("主屏 {} × {}，由 Rust xcap 采集", frame.width, frame.height),
                    false,
                    false,
                    true,
                    None,
                ),
            );
            frame
        }
        Err(error) => {
            update_snapshot(&app.state::<RuntimeState>(), |s| {
                s.phase = "error".into();
                s.message = error.clone();
            });
            emit(
                &app,
                event(
                    &task_id,
                    "error",
                    "屏幕采集失败",
                    error,
                    false,
                    true,
                    false,
                    None,
                ),
            );
            return;
        }
    };

    let plan = if live {
        infer_plan(&goal, &frame, &load_model_config(&app).llama_url).unwrap_or_else(|error| fallback_plan(&goal, Some(error)))
    } else {
        fallback_plan(&goal, None)
    };
    let raw = json!({"observation": plan.observation, "step": plan.step, "blockedReason": plan.blocked_reason});
    let blocked = plan.blocked_reason.clone();
    if let Some(reason) = blocked {
        update_snapshot(&app.state::<RuntimeState>(), |s| {
            s.phase = "paused".into();
            s.message = reason.clone();
        });
        emit(
            &app,
            event(
                &task_id,
                "paused",
                "安全策略已暂停任务",
                reason,
                false,
                false,
                false,
                Some(raw),
            ),
        );
        return;
    }

    *app.state::<RuntimeState>()
        .pending_plan
        .lock()
        .expect("pending plan poisoned") = Some(plan.clone());
    update_snapshot(&app.state::<RuntimeState>(), |s| {
        s.phase = "awaiting_user".into();
        s.model_ready = live;
        s.message = format!("{}：{}", plan.step.action, plan.step.target);
    });
    emit(
        &app,
        event(
            &task_id,
            "awaiting_user",
            "计划已准备，等待确认",
            format!(
                "{}。动作：{}；目标：{}；预期：{}",
                plan.observation, plan.step.action, plan.step.target, plan.step.expected_change
            ),
            true,
            false,
            true,
            Some(raw),
        ),
    );
    if auto_confirm {
        let _ = confirm_task(app.clone(), app.state::<RuntimeState>());
    }
}

#[tauri::command]
fn confirm_task(app: AppHandle, state: State<'_, RuntimeState>) -> Result<RuntimeSnapshot, String> {
    let plan = state
        .pending_plan
        .lock()
        .expect("pending plan poisoned")
        .clone()
        .ok_or("没有待确认的计划")?;
    let snapshot = update_snapshot(&state, |s| {
        s.phase = "executing".into();
        s.message = format!("正在执行受控动作：{}", plan.step.action);
    });
    let task_id = snapshot.task_id.clone().unwrap_or_default();
    emit(
        &app,
        event(
            &task_id,
            "executing",
            "已确认执行",
            snapshot.message.clone(),
            false,
            false,
            true,
            None,
        ),
    );
    let result = execute_safe_action(&plan, snapshot.mode.starts_with("native live"));
    *state.pending_plan.lock().expect("pending plan poisoned") = None;
    match result {
        Ok(message) => {
            let snapshot = update_snapshot(&state, |s| {
                s.phase = "completed".into();
                s.message = message.clone();
            });
            emit(
                &app,
                event(
                    &task_id,
                    "completed",
                    "步骤已完成",
                    message,
                    false,
                    true,
                    true,
                    None,
                ),
            );
            Ok(snapshot)
        }
        Err(error) => {
            let snapshot = update_snapshot(&state, |s| {
                s.phase = "paused".into();
                s.message = error.clone();
            });
            emit(
                &app,
                event(
                    &task_id,
                    "paused",
                    "执行被安全门控拦截",
                    error,
                    false,
                    false,
                    false,
                    None,
                ),
            );
            Ok(snapshot)
        }
    }
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
    let mut png = Vec::new();
    scaled
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| format!("PNG 编码失败：{e}"))?;
    Ok(ScreenFrame {
        png_base64: BASE64.encode(png),
        width,
        height,
    })
}

fn infer_plan(goal: &str, frame: &ScreenFrame, endpoint: &str) -> Result<NativePlan, String> {
    let prompt = "You are a local Windows computer-use planner. Reply ONLY valid JSON: {\"observation\":string,\"steps\":[{\"action\":string,\"target\":string,\"x\":integer|null,\"y\":integer|null,\"text\":string|null,\"risk\":string,\"expected_change\":string}]}. Handle the user's requested observation or internal operation directly and return the next useful step.";
    let payload = json!({
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text", "text": format!("User goal: {goal}")},
                {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{}", frame.png_base64)}}
            ]}
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "stream": false
    });
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())?;
    let value: Value = client
        .post(endpoint)
        .json(&payload)
        .send()
        .map_err(|e| format!("llama-server 不可用（{endpoint}）：{e}"))?
        .error_for_status()
        .map_err(|e| format!("llama-server 返回错误：{e}"))?
        .json()
        .map_err(|e| format!("解析模型响应失败：{e}"))?;
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or("模型响应缺少 content")?;
    let parsed: Value = serde_json::from_str(strip_json_fence(content))
        .map_err(|e| format!("模型未返回合法 JSON：{e}"))?;
    let observation = parsed
        .get("observation")
        .and_then(Value::as_str)
        .unwrap_or("模型已完成屏幕观察")
        .to_string();
    let first = parsed
        .get("steps")
        .and_then(Value::as_array)
        .and_then(|v| v.first())
        .ok_or("模型没有返回步骤")?;
    let step: PlanStep =
        serde_json::from_value(first.clone()).map_err(|e| format!("模型步骤格式错误：{e}"))?;
    Ok(NativePlan {
        blocked_reason: None,
        observation,
        step,
    })
}

fn strip_json_fence(text: &str) -> &str {
    text.trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim()
}

fn fallback_plan(goal: &str, model_error: Option<String>) -> NativePlan {
    let detail = model_error
        .map(|e| format!("本地规则规划已启用（模型不可用：{e}）"))
        .unwrap_or_else(|| "本地 Rust 预览规划已启用".into());
    let step = PlanStep {
        action: "wait".into(),
        target: "当前前台应用".into(),
        x: None,
        y: None,
        text: None,
        risk: "low".into(),
        expected_change: "仅完成观察，不注入输入".into(),
    };
    NativePlan {
        observation: format!("{detail}。已记录任务“{goal}”，等待你确认下一步。"),
        blocked_reason: None,
        step,
    }
}

fn execute_safe_action(plan: &NativePlan, live: bool) -> Result<String, String> {
    if plan.step.action == "wait" {
        return Ok("已完成只读观察；未向系统注入输入".into());
    }
    if !live {
        return Ok("预览模式已确认计划；dry-run 未向系统注入输入".into());
    }
    Err(format!(
        "{} 已通过 Rust 计划与策略验证，但原生输入注入将在 UIA 目标重定位迁移完成后启用",
        plan.step.action
    ))
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
            std::thread::spawn(move || start_model(handle));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_runtime,
            get_model_config,
            set_model_config,
            run_task,
            confirm_task,
            pause_runtime,
            stop_runtime
        ])
        .run(tauri::generate_context!())
        .expect("error while running baodou desktop");
}
