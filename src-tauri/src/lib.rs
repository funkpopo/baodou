use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use image::{imageops::FilterType, DynamicImage, ImageFormat};
use reqwest::blocking::Client;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    env,
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};
use tauri::{
    AppHandle, Emitter, LogicalPosition, Manager, Position, State, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder,
};
use uuid::Uuid;
use xcap::Monitor;

const PROTOCOL_VERSION: &str = "2.0.0";
const LLAMA_ENDPOINT: &str = "http://127.0.0.1:8765/v1/chat/completions";
const SAMPLE_INTERVAL: Duration = Duration::from_millis(900);
const DEFAULT_GOAL: &str = "描述当前屏幕上的关键可见内容";
const FLOATING_LABEL: &str = "floating";
/// Compact transparent pet window: speech bubble sits beside the spirit.
const FLOATING_WIDTH: f64 = 268.0;
const FLOATING_HEIGHT: f64 = 148.0;

struct RuntimeState {
    snapshot: Mutex<RuntimeSnapshot>,
}
struct ModelState {
    process: Mutex<Option<Child>>,
}
impl Default for RuntimeState {
    fn default() -> Self {
        Self {
            snapshot: Mutex::new(RuntimeSnapshot::default()),
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
        let root = asset_root();
        Self {
            server_path: first_existing(&[
                root.join("llama-server.exe"),
                PathBuf::from(r"D:\llama\llama-server.exe"),
            ])
            .to_string_lossy()
            .into(),
            model_path: root
                .join("model")
                .join("Qwen3.5-2B-UD-Q4_K_XL.gguf")
                .to_string_lossy()
                .into(),
            mmproj_path: root
                .join("model")
                .join("mmproj-F16.gguf")
                .to_string_lossy()
                .into(),
            llama_url: env::var("BAODOU_LLAMA_URL").unwrap_or_else(|_| LLAMA_ENDPOINT.into()),
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
            mode: "live screen recognition".into(),
            phase: "idle".into(),
            connected: true,
            inference_backend: "llama.cpp · local vision".into(),
            device: "SYCL0 · Intel Arc".into(),
            model_ready: false,
            task_id: None,
            goal: None,
            message: "本地屏幕识别运行时已就绪".into(),
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TaskRequest {
    goal: Option<String>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RecognitionEvent {
    task_id: String,
    phase: String,
    title: String,
    detail: String,
    timestamp: String,
    requires_confirmation: bool,
    complete: bool,
    ok: bool,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct FloatingMessage {
    text: String,
    phase: String,
    updated_at: String,
}

struct ScreenFrame {
    png_base64: String,
}

/// Extract text from the response shapes used by OpenAI-compatible vision
/// servers.  In particular, newer servers may return `content` as a list of
/// typed parts instead of one string.
fn response_text(value: &serde_json::Value) -> Option<String> {
    fn collect_text(value: &serde_json::Value, output: &mut Vec<String>) {
        match value {
            serde_json::Value::String(text) => {
                let text = text.trim();
                if !text.is_empty() {
                    output.push(text.to_owned());
                }
            }
            serde_json::Value::Array(parts) => {
                for part in parts {
                    // OpenAI Responses-style parts can use either `text` or
                    // `content`; recurse so both forms remain supported.
                    if let Some(text) = part.get("text") {
                        collect_text(text, output);
                    } else if let Some(content) = part.get("content") {
                        collect_text(content, output);
                    }
                }
            }
            serde_json::Value::Object(object) => {
                if let Some(text) = object.get("text") {
                    collect_text(text, output);
                } else if let Some(content) = object.get("content") {
                    collect_text(content, output);
                }
            }
            _ => {}
        }
    }

    let mut pieces = Vec::new();
    for candidate in [
        value.pointer("/choices/0/message/content"),
        value.pointer("/choices/0/text"),
        value.pointer("/output_text"),
        value.pointer("/output/0/content"),
    ]
    .into_iter()
    .flatten()
    {
        collect_text(candidate, &mut pieces);
    }

    let text = pieces.join("\n");
    (!text.is_empty()).then_some(text)
}

fn response_diagnostic(value: &serde_json::Value) -> String {
    if let Some(message) = value
        .pointer("/error/message")
        .and_then(|item| item.as_str())
    {
        return format!("模型返回错误：{message}");
    }

    let finish_reason = value
        .pointer("/choices/0/finish_reason")
        .and_then(|item| item.as_str())
        .unwrap_or("未知");
    format!("模型已收到屏幕截图，但没有生成识别内容（结束原因：{finish_reason}）")
}

#[cfg(test)]
mod response_tests {
    use super::response_text;
    use serde_json::json;

    #[test]
    fn reads_openai_string_content() {
        let response = json!({
            "choices": [{ "message": { "content": "屏幕上显示一个浏览器窗口。" } }]
        });
        assert_eq!(
            response_text(&response).as_deref(),
            Some("屏幕上显示一个浏览器窗口。")
        );
    }

    #[test]
    fn reads_structured_vision_content() {
        let response = json!({
            "choices": [{
                "message": {
                    "content": [
                        { "type": "output_text", "text": "屏幕上有一个对话框。" }
                    ]
                }
            }]
        });
        assert_eq!(
            response_text(&response).as_deref(),
            Some("屏幕上有一个对话框。")
        );
    }
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SessionHistory {
    id: String,
    goal: String,
    latest_result: String,
    started_at: String,
    updated_at: String,
}

fn now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis().to_string())
        .unwrap_or_default()
}

fn update_snapshot(state: &RuntimeState, update: impl FnOnce(&mut RuntimeSnapshot)) -> RuntimeSnapshot {
    let mut snapshot = state.snapshot.lock().expect("runtime poisoned");
    update(&mut snapshot);
    snapshot.clone()
}

fn emit_recognition(app: &AppHandle, event: RecognitionEvent) {
    let _ = app.emit("recognition-event", event);
}

fn emit_floating(app: &AppHandle, text: impl Into<String>, phase: impl Into<String>) {
    let _ = app.emit(
        "floating-message",
        FloatingMessage {
            text: text.into(),
            phase: phase.into(),
            updated_at: now(),
        },
    );
}

/// Portable root: directory containing the running executable.
/// Database and config always live under this tree.
fn portable_root() -> PathBuf {
    env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .or_else(|| env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Asset root prefers the portable exe directory, then the project tree in dev builds.
fn asset_root() -> PathBuf {
    let portable = portable_root();
    let portable_model = portable.join("model").join("Qwen3.5-2B-UD-Q4_K_XL.gguf");
    if portable_model.exists() {
        return portable;
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(project_root) = manifest_dir.parent() {
        let project_model = project_root.join("model").join("Qwen3.5-2B-UD-Q4_K_XL.gguf");
        if project_model.exists() {
            return project_root.to_path_buf();
        }
    }

    portable
}

fn first_existing(paths: &[PathBuf]) -> PathBuf {
    paths
        .iter()
        .find(|path| path.exists())
        .cloned()
        .unwrap_or_else(|| paths[0].clone())
}

fn data_dir() -> Result<PathBuf, String> {
    let dir = portable_root().join("data");
    fs::create_dir_all(&dir).map_err(|e| format!("无法创建便携数据目录：{e}"))?;
    Ok(dir)
}

fn config_path() -> Result<PathBuf, String> {
    Ok(data_dir()?.join("config.json"))
}

fn database_path() -> Result<PathBuf, String> {
    Ok(data_dir()?.join("baodou.db"))
}

fn database() -> Result<Connection, String> {
    let db = Connection::open(database_path()?).map_err(|e| format!("无法打开本地数据库：{e}"))?;
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY NOT NULL,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY NOT NULL,
            goal TEXT NOT NULL,
            latest_result TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );",
    )
    .map_err(|e| format!("无法初始化本地数据库：{e}"))?;
    Ok(db)
}

fn load_model_config() -> ModelConfig {
    if let Ok(path) = config_path() {
        if let Ok(raw) = fs::read_to_string(&path) {
            if let Ok(config) = serde_json::from_str::<ModelConfig>(&raw) {
                return config;
            }
        }
    }

    if let Ok(db) = database() {
        if let Ok(value) = db.query_row(
            "SELECT value FROM settings WHERE key = 'model_config'",
            [],
            |row| row.get::<_, String>(0),
        ) {
            if let Ok(config) = serde_json::from_str::<ModelConfig>(&value) {
                let _ = persist_config_file(&config);
                return config;
            }
        }
    }

    let config = ModelConfig::default();
    let _ = persist_config_file(&config);
    config
}

fn persist_config_file(config: &ModelConfig) -> Result<(), String> {
    let path = config_path()?;
    let raw = serde_json::to_string_pretty(config).map_err(|e| e.to_string())?;
    fs::write(&path, raw).map_err(|e| format!("配置文件写入失败：{e}"))
}

fn persist_model_config(config: &ModelConfig) -> Result<(), String> {
    persist_config_file(config)?;
    database()?.execute(
        "INSERT INTO settings (key, value) VALUES ('model_config', ?1)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![serde_json::to_string(config).map_err(|e| e.to_string())?],
    )
    .map_err(|e| format!("配置保存失败：{e}"))?;
    Ok(())
}

#[tauri::command]
fn get_model_config() -> ModelConfig {
    load_model_config()
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

    let config = ModelConfig {
        server_path: config.server_path.trim().into(),
        model_path: config.model_path.trim().into(),
        mmproj_path: config.mmproj_path.trim().into(),
        llama_url: config.llama_url.trim().into(),
    };
    persist_model_config(&config)?;
    stop_model(&app);
    let app_clone = app.clone();
    std::thread::spawn(move || start_model(app_clone));
    Ok(config)
}

fn stop_model(app: &AppHandle) {
    if let Some(mut child) = app
        .state::<ModelState>()
        .process
        .lock()
        .expect("model poisoned")
        .take()
    {
        let _ = child.kill();
        let _ = child.wait();
    }
}

fn llama_health(endpoint: &str) -> bool {
    let url = endpoint
        .split("/v1/")
        .next()
        .map(|s| format!("{s}/health"))
        .unwrap_or_default();
    Client::builder()
        .timeout(Duration::from_millis(600))
        .build()
        .ok()
        .and_then(|client| client.get(url).send().ok())
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn start_model(app: AppHandle) {
    let config = load_model_config();
    if llama_health(&config.llama_url) {
        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
            snapshot.model_ready = true;
            snapshot.message = "本地视觉模型已连接".into();
        });
        return;
    }

    let server = PathBuf::from(&config.server_path);
    let model = PathBuf::from(&config.model_path);
    let mmproj = PathBuf::from(&config.mmproj_path);
    if !server.exists() || !model.exists() || !mmproj.exists() {
        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
            snapshot.message = format!(
                "本地模型未就绪：请将 llama-server 与 model 放在 {}",
                portable_root().display()
            );
        });
        return;
    }

    let host_port = config
        .llama_url
        .split("://")
        .nth(1)
        .unwrap_or("127.0.0.1:8765")
        .split('/')
        .next()
        .unwrap_or("127.0.0.1:8765");
    let mut parts = host_port.rsplitn(2, ':');
    let port = parts.next().unwrap_or("8765");
    let host = parts.next().unwrap_or("127.0.0.1");

    match Command::new(&server)
        .current_dir(server.parent().unwrap_or_else(|| Path::new(".")))
        .args([
            "-m",
            model.to_string_lossy().as_ref(),
            "--mmproj",
            mmproj.to_string_lossy().as_ref(),
            "--host",
            host,
            "--port",
            port,
            "--jinja",
            "-c",
            // Vision tokens for a screen image plus a short answer need more
            // room than the former 2K context.  A too-small context can make
            // llama.cpp stop before it reaches the answer.
            "4096",
            "--threads",
            "6",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(child) => {
            *app.state::<ModelState>()
                .process
                .lock()
                .expect("model poisoned") = Some(child);
            for _ in 0..45 {
                if llama_health(&config.llama_url) {
                    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                        snapshot.model_ready = true;
                        snapshot.message = "本地视觉模型已就绪".into();
                    });
                    return;
                }
                std::thread::sleep(Duration::from_millis(400));
            }
        }
        Err(error) => {
            update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                snapshot.message = format!("模型启动失败：{error}");
            });
        }
    }
}

fn capture_primary() -> Result<ScreenFrame, String> {
    let monitor = Monitor::all()
        .map_err(|e| format!("枚举显示器失败：{e}"))?
        .into_iter()
        .find(|m| m.is_primary().unwrap_or(false))
        .ok_or_else(|| "没有检测到主显示器".to_string())?;
    let image = monitor
        .capture_image()
        .map_err(|e| format!("屏幕采集失败：{e}"))?;
    let scaled = DynamicImage::ImageRgba8(image).resize(960, 540, FilterType::Triangle);
    let mut png = Vec::new();
    scaled
        .write_to(&mut Cursor::new(&mut png), ImageFormat::Png)
        .map_err(|e| format!("图像编码失败：{e}"))?;
    Ok(ScreenFrame {
        png_base64: BASE64.encode(png),
    })
}

fn recognize(endpoint: &str, query: &str, frame: &ScreenFrame) -> Result<String, String> {
    let prompt = format!(
        "/no_think\n你是实时屏幕识别助手。此消息附有一张刚刚截取的当前屏幕 PNG 图片，必须直接观察这张图片作答，不能只根据文字猜测。不要展示思考过程，直接给出最终识别结果。用户关注：{query}\n仅描述当前画面中与关注点最相关的可见信息。不要建议点击、输入、打开、关闭或操作任何程序；不要输出步骤、坐标或 ACTION。用简洁中文回答，最多三句、120字。若图片不清晰或无法判断，直接说明原因。"
    );
    let payload = json!({
        "model": "local-vision",
        // Qwen can otherwise spend the entire 140-token budget in its hidden
        // reasoning phase, leaving `content` empty and returning `length`.
        "temperature": 0.1,
        "max_tokens": 384,
        "reasoning_format": "none",
        "chat_template_kwargs": { "enable_thinking": false },
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": prompt },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/png;base64,{}", frame.png_base64)
                    }
                }
            ]
        }]
    });

    let response: serde_json::Value = Client::builder()
        .timeout(Duration::from_secs(25))
        .build()
        .map_err(|e| e.to_string())?
        .post(endpoint)
        .json(&payload)
        .send()
        .map_err(|e| format!("识别请求失败：{e}"))?
        .error_for_status()
        .map_err(|e| format!("模型接口错误：{e}"))?
        .json()
        .map_err(|e| format!("模型响应解析失败：{e}"))?;

    response_text(&response).ok_or_else(|| response_diagnostic(&response))
}

fn task_active(app: &AppHandle, id: &str) -> bool {
    let snapshot = app
        .state::<RuntimeState>()
        .snapshot
        .lock()
        .expect("runtime poisoned")
        .clone();
    snapshot.task_id.as_deref() == Some(id) && snapshot.phase == "recognizing"
}

#[tauri::command]
fn get_session_history() -> Result<Vec<SessionHistory>, String> {
    let db = database()?;
    let mut statement = db
        .prepare(
            "SELECT id, goal, latest_result, started_at, updated_at
             FROM sessions
             ORDER BY updated_at DESC
             LIMIT 100",
        )
        .map_err(|e| e.to_string())?;
    let rows = statement
        .query_map([], |row| {
            Ok(SessionHistory {
                id: row.get(0)?,
                goal: row.get(1)?,
                latest_result: row.get(2)?,
                started_at: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn get_runtime(state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    state.snapshot.lock().expect("runtime poisoned").clone()
}

#[tauri::command]
fn get_portable_paths() -> Result<serde_json::Value, String> {
    Ok(json!({
        "root": portable_root().to_string_lossy(),
        "dataDir": data_dir()?.to_string_lossy(),
        "configPath": config_path()?.to_string_lossy(),
        "databasePath": database_path()?.to_string_lossy(),
    }))
}

#[tauri::command]
fn run_task(app: AppHandle, state: State<'_, RuntimeState>, request: TaskRequest) -> Result<String, String> {
    let goal = request
        .goal
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_GOAL)
        .to_string();

    let id = Uuid::new_v4().to_string();
    let timestamp = now();
    database()?.execute(
        "INSERT INTO sessions (id, goal, started_at, updated_at) VALUES (?1, ?2, ?3, ?3)",
        params![id, goal, timestamp],
    )
    .map_err(|e| format!("无法保存会话：{e}"))?;

    update_snapshot(&state, |snapshot| {
        snapshot.mode = "live screen recognition".into();
        snapshot.phase = "recognizing".into();
        snapshot.task_id = Some(id.clone());
        snapshot.goal = Some(goal.clone());
        snapshot.message = "正在采集屏幕并识别".into();
    });

    show_floating_window(app.clone()).map_err(|e| format!("无法显示悬浮窗：{e}"))?;
    emit_floating(&app, "正在识别屏幕内容…", "recognizing");

    let clone = app.clone();
    std::thread::spawn(move || recognition_loop(clone, id, goal));
    Ok("started".into())
}

fn recognition_loop(app: AppHandle, task_id: String, goal: String) {
    let endpoint = load_model_config().llama_url;
    let mut last = String::new();

    while task_active(&app, &task_id) {
        let started = Instant::now();
        match capture_primary().and_then(|frame| recognize(&endpoint, &goal, &frame)) {
            Ok(text) => {
                update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                    snapshot.model_ready = true;
                    snapshot.message = text.clone();
                });
                if text != last {
                    let timestamp = now();
                    if let Ok(db) = database() {
                        let _ = db.execute(
                            "UPDATE sessions SET latest_result = ?1, updated_at = ?2 WHERE id = ?3",
                            params![text, timestamp, task_id],
                        );
                    }
                    emit_recognition(
                        &app,
                        RecognitionEvent {
                            task_id: task_id.clone(),
                            phase: "recognizing".into(),
                            title: "屏幕识别结果".into(),
                            detail: text.clone(),
                            timestamp,
                            requires_confirmation: false,
                            complete: false,
                            ok: true,
                        },
                    );
                    emit_floating(&app, text.clone(), "recognizing");
                    last = text;
                }
            }
            Err(error) => {
                update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                    snapshot.message = error.clone();
                });
                emit_floating(&app, format!("识别暂不可用：{error}"), "error");
            }
        }

        let elapsed = started.elapsed();
        if elapsed < SAMPLE_INTERVAL {
            std::thread::sleep(SAMPLE_INTERVAL - elapsed);
        }
    }
}

#[tauri::command]
fn stop_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    let snapshot = update_snapshot(&state, |s| {
        s.phase = "stopped".into();
        s.message = "已停止实时屏幕识别".into();
    });
    emit_floating(&app, "识别已暂停", "stopped");
    let _ = hide_floating_window(app);
    snapshot
}

#[tauri::command]
fn pause_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    stop_runtime(app, state)
}

fn ensure_floating_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    if let Some(window) = app.get_webview_window(FLOATING_LABEL) {
        return Ok(window);
    }

    // Fallback when the preconfigured window is missing (e.g. older builds).
    // Load with an explicit query so the frontend can still detect floating mode.
    WebviewWindowBuilder::new(
        app,
        FLOATING_LABEL,
        WebviewUrl::App("index.html?window=floating".into()),
    )
    .title("baodou")
    .inner_size(FLOATING_WIDTH, FLOATING_HEIGHT)
    .min_inner_size(220.0, 120.0)
    .visible(false)
    .always_on_top(true)
    .decorations(false)
    .transparent(true)
    .shadow(false)
    .skip_taskbar(true)
    .resizable(false)
    .focused(false)
    .build()
    .map_err(|e| format!("创建悬浮窗失败：{e}"))
}

fn position_floating_window(app: &AppHandle, window: &WebviewWindow) -> Result<(), String> {
    let monitor = app
        .primary_monitor()
        .map_err(|e| e.to_string())?
        .or_else(|| app.available_monitors().ok().and_then(|list| list.into_iter().next()))
        .ok_or_else(|| "无法获取显示器信息".to_string())?;

    let scale = monitor.scale_factor();
    let screen = monitor.size();
    let work_w = screen.width as f64 / scale;
    let work_h = screen.height as f64 / scale;
    let x = (work_w - FLOATING_WIDTH - 24.0).max(12.0);
    let y = (work_h - FLOATING_HEIGHT - 56.0).max(12.0);

    window
        .set_size(tauri::Size::Logical(tauri::LogicalSize::new(
            FLOATING_WIDTH,
            FLOATING_HEIGHT,
        )))
        .map_err(|e| format!("设置悬浮窗尺寸失败：{e}"))?;
    window
        .set_position(Position::Logical(LogicalPosition::new(x, y)))
        .map_err(|e| format!("定位悬浮窗失败：{e}"))?;
    Ok(())
}

#[tauri::command]
fn show_floating_window(app: AppHandle) -> Result<(), String> {
    let window = ensure_floating_window(&app)?;
    position_floating_window(&app, &window)?;
    window
        .set_always_on_top(true)
        .map_err(|e| format!("置顶悬浮窗失败：{e}"))?;
    window
        .show()
        .map_err(|e| format!("显示悬浮窗失败：{e}"))?;
    // Keep the overlay above other windows without aggressively stealing keyboard focus.
    let _ = window.unminimize();
    Ok(())
}

#[tauri::command]
fn hide_floating_window(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window(FLOATING_LABEL) {
        window.hide().map_err(|e| format!("隐藏悬浮窗失败：{e}"))?;
    }
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeState::default())
        .manage(ModelState::default())
        .setup(|app| {
            let _ = data_dir().map_err(|e| std::io::Error::other(e))?;
            let _ = database().map_err(|e| std::io::Error::other(e))?;
            let _ = persist_config_file(&load_model_config());

            // Ensure the floating webview is created at startup (hidden),
            // so show/hide later only toggles visibility.
            if let Ok(window) = ensure_floating_window(&app.handle()) {
                let _ = position_floating_window(&app.handle(), &window);
                let _ = window.hide();
            }

            let handle = app.handle().clone();
            std::thread::spawn(move || start_model(handle));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_runtime,
            get_model_config,
            set_model_config,
            get_session_history,
            get_portable_paths,
            run_task,
            pause_runtime,
            stop_runtime,
            show_floating_window,
            hide_floating_window
        ])
        .run(tauri::generate_context!())
        .expect("error while running baodou desktop");
}
