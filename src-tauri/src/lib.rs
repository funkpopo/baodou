mod capture;
mod sampling;
mod textsim;

use capture::ScreenFrame;
use reqwest::blocking::Client;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::Mutex,
    time::{Duration, Instant},
};
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    AppHandle, Emitter, LogicalPosition, Manager, Position, State, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder, WindowEvent,
};
use uuid::Uuid;

const PROTOCOL_VERSION: &str = "2.0.0";
const LLAMA_ENDPOINT: &str = "http://[IP]:8765/v1/chat/completions";
const DEFAULT_GOAL: &str = "帮我观察当前电脑界面，留意最要紧、最清楚的可见内容";
const FLOATING_LABEL: &str = "floating";
/// Pet-only default HWND. The frontend resizes this to the visible pet +
/// bubble so leftover transparent chrome does not swallow clicks.
const FLOATING_WIDTH: f64 = 104.0;
const FLOATING_HEIGHT: f64 = 104.0;
const FLOATING_MIN_WIDTH: f64 = 104.0;
const FLOATING_MIN_HEIGHT: f64 = 104.0;
const FLOATING_MAX_WIDTH: f64 = 480.0;
const FLOATING_MAX_HEIGHT: f64 = 240.0;

// --- P0: streamed display throttle (sentence-first, then 120–200 ms) ---
const FLUSH_FIRST_CHARS: usize = 26;
const FLUSH_THROTTLE: Duration = Duration::from_millis(150);
const FLUSH_FORCE_TIMEOUT: Duration = Duration::from_millis(1600);
/// Leave enough room for a complete visual summary.  Some local vision models
/// emit additional visible context despite the concise prompt, and a 160-token
/// ceiling can cut a Chinese sentence off mid-way.
const MAX_RECOGNITION_TOKENS: u32 = 512;

// --- P1: sampling & refresh policies (in addition to `sampling` module) ---
/// Above this share of the screen a changed bbox is treated as a broad change
/// and sent immediately; below it the change is "small" and needs confirmation.
const SIGNIFICANT_AREA_FRACTION: f64 = 0.06;
/// A localised change at most this large is sent as a high-density crop.
const CROP_MAX_AREA_FRACTION: f64 = 0.30;
/// Minimum gap between two model requests to protect a small local server.
const MIN_MODEL_GAP: Duration = Duration::from_millis(600);
/// After a low-information result, small changes are ignored for this long.
const LOW_INFO_SUPPRESS: Duration = Duration::from_millis(6000);
/// After a conservative contradiction message, re-report only after this long.
const CONTRADICTION_COOLDOWN: Duration = Duration::from_millis(12000);
const TRAY_SHOW_ID: &str = "show-main";
const TRAY_EXIT_ID: &str = "quit";

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
    /// P3 server tuning knobs. They never touch the context length `-c`:
    /// only offload, batch, thread and Flash Attention parameters.
    #[serde(default)]
    n_gpu_layers: Option<i32>,
    #[serde(default)]
    batch_size: Option<i32>,
    #[serde(default)]
    ubatch_size: Option<i32>,
    #[serde(default)]
    flash_attn: bool,
    /// Opt-in single-request multi-image ("thumbnail + crop"). Off by default
    /// until multi-image input is verified stable on the deployed model.
    #[serde(default = "default_false")]
    multi_image_input: bool,
}

fn default_false() -> bool {
    false
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
            n_gpu_layers: env_i32("BAODOU_N_GPU_LAYERS"),
            batch_size: env_i32("BAODOU_BATCH_SIZE"),
            ubatch_size: env_i32("BAODOU_UBATCH_SIZE"),
            flash_attn: env_flag("BAODOU_FLASH_ATTN"),
            multi_image_input: env_flag("BAODOU_MULTI_IMAGE"),
        }
    }
}

fn env_i32(key: &str) -> Option<i32> {
    env::var(key)
        .ok()
        .and_then(|value| value.trim().parse().ok())
}

fn env_flag(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            value == "1" || value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("on")
        })
        .unwrap_or(false)
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
    /// Live counters for benchmark / acceptance runs.
    rounds: u64,
    skipped_rounds: u64,
    requests: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    metrics: Option<OpsMetrics>,
}

/// P3 per-request breakdown: where time actually goes.
#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct OpsMetrics {
    capture_ms: Option<f64>,
    encode_ms: Option<f64>,
    /// Request connect + send + prefill + first token.
    first_token_ms: Option<f64>,
    /// Remaining generation (first token … stream end).
    generate_ms: Option<f64>,
    total_ms: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    /// Readability of the last published result.
    readability: Option<String>,
    /// Whether the request used a full-screen thumbnail or a localised crop.
    input_kind: Option<String>,
    /// Summarised llama-server /metrics scrape.
    server: Option<String>,
    error: Option<String>,
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
            message: "我在桌边呢，随时可以帮你看屏幕。".into(),
            rounds: 0,
            skipped_rounds: 0,
            requests: 0,
            metrics: None,
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

/// Result of one recognition round, alongside its streamed timing breakdown.
struct RecognizeResult {
    text: String,
    first_token_ms: Option<f64>,
    generate_ms: Option<f64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
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

    let text = strip_thinking(&pieces.join("\n"));
    (!text.is_empty()).then_some(text)
}

/// Model templates occasionally expose reasoning despite `/no_think`.
/// Never show that private reasoning in the recognition UI or save it to the
/// session history. Handles mixed-case, multiline, and incomplete tags.
fn strip_thinking(text: &str) -> String {
    let mut cleaned = String::with_capacity(text.len());
    let mut rest = text;

    loop {
        let lower = rest.to_ascii_lowercase();
        let Some(start) = lower.find("<think>") else {
            cleaned.push_str(rest);
            break;
        };

        cleaned.push_str(&rest[..start]);
        let after_open = &rest[start + "<think>".len()..];
        let after_open_lower = after_open.to_ascii_lowercase();
        let Some(end) = after_open_lower.find("</think>") else {
            // An unfinished thinking block must not leak while a model is
            // stopped or truncated before its closing tag.
            break;
        };
        rest = &after_open[end + "</think>".len()..];
    }

    cleaned.trim().to_owned()
}

/// Extract one visible text delta from an OpenAI-compatible SSE chunk.
/// llama-server uses `choices[].delta.content`, but keep the alternate text
/// shape for compatible local servers as well.
fn stream_delta_text(value: &serde_json::Value) -> Option<String> {
    let delta = value
        .pointer("/choices/0/delta/content")
        .or_else(|| value.pointer("/choices/0/delta/text"))?;
    match delta {
        serde_json::Value::String(text) => (!text.is_empty()).then(|| text.to_owned()),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => response_text(&json!({
            "choices": [{ "message": { "content": delta } }]
        })),
        _ => None,
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;

    #[test]
    fn first_flush_wait_for_sentence_boundary_then_throttle() {
        let mut flusher = StreamFlusher::new();
        // Short partial text without a boundary stays silent.
        assert!(flusher.feed("正在读取").is_none());
        // A sentence boundary triggers the first display.
        let first = flusher
            .feed("正在读取系统状态。")
            .expect("flush on boundary");
        assert_eq!(first, "正在读取系统状态。");
        // Identical length (no new content) produces nothing.
        assert!(flusher.feed("正在读取系统状态。").is_none());
        // A small new chunk without a boundary is held back.
        assert!(flusher.feed("正在读取系统状态。已连接").is_none());
    }

    #[test]
    fn long_stream_without_boundary_force_flushes_with_ellipsis() {
        let mut flusher = StreamFlusher::new();
        assert!(flusher.feed("正在").is_none());
        // Enough characters without a terminator force a provisional flush.
        let long = "正在进行系统初始化并加载用户配置且正在同步远程数据窗口显示";
        let display = flusher.feed(long).expect("block flush");
        assert!(display.contains(long));
        assert!(display.ends_with('…'));
    }

    #[test]
    fn final_without_terminal_text_is_marked_provisional() {
        let mut flusher = StreamFlusher::new();
        assert!(flusher.feed("一切正常").is_none());
        let long = "一切正常没有异常需要处理请稍候等待状态更新完毕后再继续操作";
        let display = flusher.feed(long).expect("flush");
        assert!(display.contains(long));
        assert!(display.ends_with('…'));
    }
}

#[cfg(test)]
mod response_tests {
    use super::{response_text, stream_delta_text, strip_thinking};
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

    #[test]
    fn removes_complete_and_incomplete_thinking_blocks() {
        assert_eq!(
            strip_thinking("<think>先分析图片\n再组织答案</think>屏幕上有一个浏览器。"),
            "屏幕上有一个浏览器。"
        );
        assert_eq!(strip_thinking("可见结果。<THINK>不应显示"), "可见结果。");
    }

    #[test]
    fn reads_streaming_delta_content() {
        let chunk = json!({ "choices": [{ "delta": { "content": "正在生成" } }] });
        assert_eq!(stream_delta_text(&chunk).as_deref(), Some("正在生成"));
    }

    #[test]
    fn builds_health_url_from_chat_completions_endpoint() {
        assert_eq!(
            super::llama_health_url("http://127.0.0.1:8765/v1/chat/completions").as_deref(),
            Some("http://127.0.0.1:8765/health")
        );
        assert_eq!(super::llama_health_url("").as_deref(), None);
    }

    #[test]
    fn treats_local_llama_url_as_loopback() {
        assert!(super::is_loopback_endpoint(
            "http://127.0.0.1:8765/v1/chat/completions"
        ));
        assert!(super::is_loopback_endpoint(
            "http://localhost:8765/v1/chat/completions"
        ));
        assert!(!super::is_loopback_endpoint(
            "http://10.0.0.8:8765/v1/chat/completions"
        ));
    }

    #[test]
    fn loopback_health_check_ignores_http_proxy() {
        use std::{
            io::{Read, Write},
            net::TcpListener,
            thread,
        };

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback health fixture");
        let addr = listener.local_addr().expect("local addr");
        thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buffer = [0_u8; 256];
                let _ = stream.read(&mut buffer);
                let _ = stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{\"status\":\"ok\"}");
            }
        });

        let previous: Vec<(&str, Option<String>)> =
            ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]
                .into_iter()
                .map(|key| (key, std::env::var(key).ok()))
                .collect();

        std::env::set_var("HTTP_PROXY", "http://127.0.0.1:9");
        std::env::set_var("http_proxy", "http://127.0.0.1:9");
        std::env::set_var("HTTPS_PROXY", "http://127.0.0.1:9");
        std::env::set_var("https_proxy", "http://127.0.0.1:9");

        let endpoint = format!("http://{addr}/v1/chat/completions");
        let ready = super::llama_health(&endpoint);

        for (key, value) in previous {
            match value {
                Some(value) => std::env::set_var(key, value),
                None => std::env::remove_var(key),
            }
        }

        assert!(ready);
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

fn update_snapshot(
    state: &RuntimeState,
    update: impl FnOnce(&mut RuntimeSnapshot),
) -> RuntimeSnapshot {
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
        let project_model = project_root
            .join("model")
            .join("Qwen3.5-2B-UD-Q4_K_XL.gguf");
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
    database()?
        .execute(
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
        n_gpu_layers: config.n_gpu_layers,
        batch_size: config.batch_size,
        ubatch_size: config.ubatch_size,
        flash_attn: config.flash_attn,
        multi_image_input: config.multi_image_input,
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

fn is_loopback_endpoint(endpoint: &str) -> bool {
    let host_port = endpoint
        .split("://")
        .nth(1)
        .unwrap_or(endpoint)
        .split('/')
        .next()
        .unwrap_or("");
    host_port.starts_with("127.0.0.1")
        || host_port.starts_with("localhost")
        || host_port.starts_with("[::1]")
        || host_port == "::1"
}

/// Loopback llama-server traffic must never inherit HTTP(S)_PROXY. A system
/// proxy on 127.0.0.1 (common with clash/v2ray) returns 502 for
/// `http://127.0.0.1:8765/health` and leaves the UI stuck on 模型未就绪.
fn local_http_client(endpoint: &str, timeout: Duration) -> Result<Client, reqwest::Error> {
    let mut builder = Client::builder().timeout(timeout);
    if is_loopback_endpoint(endpoint) {
        builder = builder.no_proxy();
    }
    builder.build()
}

fn llama_health_url(endpoint: &str) -> Option<String> {
    endpoint
        .split("/v1/")
        .next()
        .filter(|base| !base.is_empty())
        .map(|base| format!("{base}/health"))
}

fn llama_health(endpoint: &str) -> bool {
    let Some(url) = llama_health_url(endpoint) else {
        return false;
    };
    local_http_client(endpoint, Duration::from_secs(2))
        .ok()
        .and_then(|client| client.get(url).send().ok())
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn mark_model_ready(app: &AppHandle, message: &str) {
    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.model_ready = true;
        snapshot.message = message.into();
    });
}

fn mark_model_not_ready(app: &AppHandle, message: impl Into<String>) {
    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.model_ready = false;
        snapshot.message = message.into();
    });
}

fn model_process_alive(app: &AppHandle) -> bool {
    let state = app.state::<ModelState>();
    let mut guard = state.process.lock().expect("model poisoned");
    match guard.as_mut() {
        Some(child) => match child.try_wait() {
            Ok(None) => true,
            Ok(Some(_)) => {
                *guard = None;
                false
            }
            Err(_) => false,
        },
        None => false,
    }
}

fn wait_for_model(app: &AppHandle, endpoint: &str, attempts: u32) -> bool {
    for _ in 0..attempts {
        if llama_health(endpoint) {
            mark_model_ready(app, "本地视觉模型已就绪");
            return true;
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    false
}

fn start_model(app: AppHandle) {
    let config = load_model_config();
    if llama_health(&config.llama_url) {
        mark_model_ready(&app, "本地视觉模型已连接");
        return;
    }

    if model_process_alive(&app) {
        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
            if !snapshot.model_ready {
                snapshot.message = "正在等待本地视觉模型就绪…".into();
            }
        });
        let _ = wait_for_model(&app, &config.llama_url, 90);
        return;
    }

    let server = PathBuf::from(&config.server_path);
    let model = PathBuf::from(&config.model_path);
    let mmproj = PathBuf::from(&config.mmproj_path);
    if !server.exists() || !model.exists() || !mmproj.exists() {
        mark_model_not_ready(
            &app,
            format!(
                "本地模型未就绪：请将 llama-server 与 model 放在 {}",
                portable_root().display()
            ),
        );
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

    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.model_ready = false;
        snapshot.message = "正在启动本地视觉模型…".into();
    });

    let mut args = vec![
        "-m".to_string(),
        model.to_string_lossy().into_owned(),
        "--mmproj".to_string(),
        mmproj.to_string_lossy().into_owned(),
        "--host".to_string(),
        host.to_string(),
        "--port".to_string(),
        port.to_string(),
        "--jinja".to_string(),
        // Context length is intentionally left at 4096: the tuning pass below
        // must not modify `-c`.  A too-small context can make llama.cpp stop
        // before producing an answer, so keep this value stable.
        "-c".to_string(),
        "4096".to_string(),
        "--threads".to_string(),
        "6".to_string(),
    ];
    // P3 server tuning knobs: GPU offload, batch sizes and Flash Attention.
    if let Some(ngl) = config.n_gpu_layers {
        args.push("-ngl".to_string());
        args.push(ngl.to_string());
    }
    if let Some(batch) = config.batch_size {
        args.push("-b".to_string());
        args.push(batch.to_string());
    }
    if let Some(batch) = config.ubatch_size {
        args.push("-ub".to_string());
        args.push(batch.to_string());
    }
    if config.flash_attn {
        args.push("-fa".to_string());
    }

    match Command::new(&server)
        .current_dir(server.parent().unwrap_or_else(|| Path::new(".")))
        .args(&args)
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
            if !wait_for_model(&app, &config.llama_url, 90) {
                mark_model_not_ready(&app, "本地模型启动超时：推理服务未在预期时间内就绪");
            }
        }
        Err(error) => {
            mark_model_not_ready(&app, format!("模型启动失败：{error}"));
        }
    }
}

fn supervise_model(app: AppHandle) {
    start_model(app.clone());
    loop {
        std::thread::sleep(Duration::from_secs(2));
        let config = load_model_config();
        if llama_health(&config.llama_url) {
            update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                if !snapshot.model_ready {
                    snapshot.model_ready = true;
                    snapshot.message = "本地视觉模型已就绪".into();
                }
            });
            continue;
        }

        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
            if snapshot.model_ready {
                snapshot.model_ready = false;
                snapshot.message = "本地视觉模型连接已断开，正在重试…".into();
            }
        });
        start_model(app.clone());
    }
}

/// Scrapes llama.cpp's Prometheus `/metrics` endpoint for the performance
/// counters most useful to the P3 benchmark pass: prompt eval, token eval,
/// cache hits and KV usage.  Only a short summary is kept in memory; the raw
/// endpoint name list is deliberately version-tolerant.
fn server_metrics(endpoint: &str) -> Option<String> {
    let base = endpoint.split("/v1/").next()?;
    if base.is_empty() {
        return None;
    }
    let url = format!("{base}/metrics");
    let text = local_http_client(endpoint, Duration::from_secs(3))
        .ok()?
        .get(&url)
        .send()
        .ok()?
        .error_for_status()
        .ok()?
        .text()
        .ok()?;
    const INTERESTING: &[&str] = &[
        "n_prompt_tokens",
        "n_predicted",
        "prompt_per_second",
        "predicted_per_second",
        "prompt_cache",
        "cache_hit",
        "kv_cache_usage",
        "kv_cache_tokens",
        "t_prompt",
        "t_predict",
        "n_prompt_eval",
        "slot_processing",
    ];
    let mut summary = Vec::with_capacity(12);
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let name = line.split_whitespace().next().unwrap_or_default();
        if INTERESTING
            .iter()
            .any(|key| name.to_ascii_lowercase().contains(key))
        {
            summary.push(name.to_string());
            if summary.len() >= 12 {
                break;
            }
        }
    }
    if summary.is_empty() {
        return None;
    }
    Some(summary.join(", "))
}

fn poll_server_metrics(app: AppHandle) {
    loop {
        std::thread::sleep(Duration::from_secs(3));
        let endpoint = load_model_config().llama_url;
        if !llama_health(&endpoint) {
            continue;
        }
        let Some(summary) = server_metrics(&endpoint) else {
            continue;
        };
        update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
            let metrics = snapshot.metrics.get_or_insert_default();
            metrics.server = Some(summary);
        });
        let _ = app.emit("recognition-metrics", &());
    }
}

/// P0 streamed display throttle: instead of forwarding every token delta, the
/// bubble only updates on a sentence boundary, on a ~26 char block, on a
/// 150 ms cadence, or when a force-flush timeout fires for long sentences.
/// The final (uncorrected) result is always sent once by the caller.
struct StreamFlusher {
    flushed: usize,
    last_flush: Instant,
    last_force: Instant,
    first: bool,
}

impl StreamFlusher {
    fn new() -> Self {
        Self {
            flushed: 0,
            last_flush: Instant::now(),
            last_force: Instant::now(),
            first: true,
        }
    }

    /// Feeds the latest cleaned text; returns the display text to render when
    /// the throttling policy says it is time (or `None` to stay silent).
    fn feed(&mut self, text: &str) -> Option<String> {
        let total = text.chars().count();
        if total <= self.flushed {
            return None;
        }
        let new_chars = total - self.flushed;
        let new_text: String = text.chars().skip(self.flushed).collect();
        let now = Instant::now();

        let boundary = has_sentence_boundary(&new_text);
        let flush = boundary
            || (new_chars >= FLUSH_FIRST_CHARS
                && (self.first || now.duration_since(self.last_flush) >= FLUSH_THROTTLE))
            || (now.duration_since(self.last_force) >= FLUSH_FORCE_TIMEOUT && new_chars > 0);
        if !flush {
            return None;
        }

        let provisional = !boundary && !ends_with_terminal(text);
        let mut display = text.to_string();
        if provisional {
            display.push('…');
        }
        self.flushed = total;
        self.last_flush = now;
        self.last_force = now;
        self.first = false;
        Some(display)
    }
}

fn has_sentence_boundary(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(
            c,
            '。' | '！' | '？' | '!' | '?' | '；' | ';' | '\n' | '\r' | '…'
        )
    })
}

fn ends_with_terminal(text: &str) -> bool {
    text.chars().last().is_some_and(|c| {
        matches!(
            c,
            '。' | '！' | '？' | '!' | '?' | '；' | ';' | '…' | '\n' | '\r'
        )
    })
}

fn recognition_prompt(query: &str) -> String {
    format!(
        "/no_think\n你是 Baodou，坐在用户桌边的拟人视觉助手。你此刻正看着眼前这张刚截取的屏幕画面，任务是陪用户观察电脑界面：把你真正看见的东西，用身边人轻声提醒的口吻说出来。用户当前想让你留意：{query}\n\
        回答要求：\n\
        1. 用第一人称短句（如“我看见…”“这边是…”），像陪在旁边看屏幕，不要写成检测报告、列表或系统日志。\n\
        2. 第一句点出画面上最重要且最明确的可见内容或界面状态。\n\
        3. 第二句再补 1–2 项与当前关注点直接相关的细节（窗口、关键文字、按钮状态、报错等）。\n\
        4. 只依据这一张截图；你看不见前后过程，不要说“变了/刚刚/正在变化”。\n\
        5. 模糊、小字、图标或数字无法确认时，如实说“我看不清/无法确认”，严禁补全、猜测或推测用户意图。\n\
        6. 你只负责看和说：禁止点击、输入、打开、关闭等操作建议，禁止输出坐标或 ACTION。\n\
        最多两句、短而完整的中文。"
    )
}

fn recognize(
    endpoint: &str,
    query: &str,
    frame: &ScreenFrame,
    mut on_delta: impl FnMut(&str),
) -> Result<RecognizeResult, String> {
    let prompt = format!("{}\n当前截图：", recognition_prompt(query));
    let mut content = Vec::with_capacity(frame.images.len() + 1);
    content.push(json!({ "type": "text", "text": prompt }));
    for image in &frame.images {
        content.push(json!({
            "type": "image_url",
            "image_url": { "url": format!("{}{}", image.mime, image.base64) }
        }));
    }
    let payload = json!({
        "model": "local-vision",
        // Qwen can otherwise spend its visible output budget in hidden
        // reasoning phase, leaving `content` empty and returning `length`.
        "temperature": 0.1,
        // The prompt asks for a concise answer, but do not use a ceiling so
        // tight that a model ends a visible sentence half-way through.
        "max_tokens": MAX_RECOGNITION_TOKENS,
        // The static instruction prefix is reused on every frame by llama.cpp.
        "cache_prompt": true,
        "stream": true,
        "reasoning_format": "none",
        "chat_template_kwargs": { "enable_thinking": false },
        "messages": [ { "role": "user", "content": content } ]
    });

    let send_started = Instant::now();
    let response = local_http_client(endpoint, Duration::from_secs(12))
        .map_err(|e| e.to_string())?
        .post(endpoint)
        .json(&payload)
        .send()
        .map_err(|e| format!("识别请求失败：{e}"))?
        .error_for_status()
        .map_err(|e| format!("模型接口错误：{e}"))?;

    let mut flusher = StreamFlusher::new();
    let mut raw_text = String::new();
    let mut first_byte: Option<Instant> = None;
    let mut prompt_tokens = None;
    let mut completion_tokens = None;
    for line in BufReader::new(response).lines() {
        let line = line.map_err(|e| format!("读取流式模型响应失败：{e}"))?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            break;
        }
        if first_byte.is_none() {
            first_byte = Some(Instant::now());
        }
        let chunk: serde_json::Value =
            serde_json::from_str(data).map_err(|e| format!("模型流式响应解析失败：{e}"))?;
        if let Some(delta) = stream_delta_text(&chunk) {
            raw_text.push_str(&delta);
        }
        if let Some(value) = chunk
            .pointer("/usage/prompt_tokens")
            .and_then(|v| v.as_u64())
        {
            prompt_tokens = Some(value);
        }
        if let Some(value) = chunk
            .pointer("/usage/completion_tokens")
            .and_then(|v| v.as_u64())
        {
            completion_tokens = Some(value);
        }
        let cleaned = strip_thinking(&raw_text);
        if let Some(display) = flusher.feed(&cleaned) {
            on_delta(&display);
        }
    }

    let final_text = strip_thinking(&raw_text);
    // The final result is always delivered once as a correction pass, even if
    // the throttled flusher already emitted a provisional (or identical) blob.
    if !final_text.is_empty() {
        on_delta(&final_text);
    }

    let done = Instant::now();
    let first_token_ms = first_byte.map(|t| t.duration_since(send_started).as_secs_f64() * 1000.0);
    let generate_ms = first_byte
        .map(|t| done.duration_since(t).as_secs_f64() * 1000.0)
        .or_else(|| Some(done.duration_since(send_started).as_secs_f64() * 1000.0));

    (!final_text.is_empty())
        .then_some(RecognizeResult {
            text: final_text,
            first_token_ms,
            generate_ms,
            prompt_tokens,
            completion_tokens,
        })
        .ok_or_else(|| "模型已收到屏幕截图，但没有生成识别内容".to_string())
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
fn run_task(
    app: AppHandle,
    state: State<'_, RuntimeState>,
    request: TaskRequest,
) -> Result<String, String> {
    let goal = request
        .goal
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_GOAL)
        .to_string();

    let id = Uuid::new_v4().to_string();
    let timestamp = now();
    database()?
        .execute(
            "INSERT INTO sessions (id, goal, started_at, updated_at) VALUES (?1, ?2, ?3, ?3)",
            params![id, goal, timestamp],
        )
        .map_err(|e| format!("无法保存会话：{e}"))?;

    update_snapshot(&state, |snapshot| {
        snapshot.mode = "live screen recognition".into();
        snapshot.phase = "recognizing".into();
        snapshot.task_id = Some(id.clone());
        snapshot.goal = Some(goal.clone());
        snapshot.message = "我先看一眼现在的屏幕…".into();
    });
    show_floating_window(app.clone()).map_err(|e| format!("无法显示悬浮窗：{e}"))?;
    let clone = app.clone();
    std::thread::spawn(move || recognition_loop(clone, id, goal));
    Ok("started".into())
}

struct SmallCandidate {
    /// The fine cells that changed.
    cells: Vec<usize>,
    /// Consecutive frames with the same small change.
    frames: u32,
    seen_at: Instant,
}

impl SmallCandidate {
    fn new(cells: Vec<usize>, now: Instant) -> Self {
        Self {
            cells,
            frames: 1,
            seen_at: now,
        }
    }
}

fn same_cell_set(left: &[usize], right: &[usize]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut sorted_left = left.to_vec();
    let mut sorted_right = right.to_vec();
    sorted_left.sort_unstable();
    sorted_right.sort_unstable();
    sorted_left == sorted_right
}

fn wait_until(started: Instant, target: Duration) {
    let elapsed = started.elapsed();
    if elapsed < target {
        std::thread::sleep(target - elapsed);
    }
}

fn count_round(app: &AppHandle) {
    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.rounds = snapshot.rounds.saturating_add(1);
    });
}

fn count_skipped(app: &AppHandle) {
    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.skipped_rounds = snapshot.skipped_rounds.saturating_add(1);
    });
}

fn count_request(app: &AppHandle) {
    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        snapshot.requests = snapshot.requests.saturating_add(1);
    });
}

fn persist_latest_result(task_id: &str, text: &str) {
    let timestamp = now();
    if let Ok(db) = database() {
        let _ = db.execute(
            "UPDATE sessions SET latest_result = ?1, updated_at = ?2 WHERE id = ?3",
            params![text, timestamp, task_id],
        );
    }
}

fn readability_label(kind: textsim::Readability) -> &'static str {
    match kind {
        textsim::Readability::Clear => "清晰可读",
        textsim::Readability::Partial => "局部可读",
        textsim::Readability::Unclear => "不宜判断",
    }
}

fn record_metrics(app: &AppHandle, mut metrics: OpsMetrics) {
    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
        let previous_server = snapshot.metrics.as_ref().and_then(|m| m.server.clone());
        metrics.server = previous_server;
        snapshot.metrics = Some(metrics.clone());
    });
    let _ = app.emit("recognition-metrics", &metrics);
}

fn recognition_loop(app: AppHandle, task_id: String, goal: String) {
    use capture::{area_fraction, capture_primary, change_bbox};
    use sampling::Motion;

    let config = load_model_config();
    let endpoint = config.llama_url;
    let multi_image = config.multi_image_input;

    let mut sampler = sampling::AdaptiveSampler::default();
    let mut last_text = String::new();
    let mut previous_coarse: Option<Vec<u64>> = None;
    let mut previous_fine: Option<Vec<u64>> = None;
    let mut small_candidate: Option<SmallCandidate> = None;
    let mut last_request_at = Instant::now() - Duration::from_secs(3600);
    let mut low_info_suppress_until = Instant::now();
    let mut contradiction_suppress_until = Instant::now();

    while task_active(&app, &task_id) {
        let frame_started = Instant::now();

        let capture_started = Instant::now();
        let captured = match capture_primary() {
            Ok(frame) => frame,
            Err(error) => {
                update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                    snapshot.message = error.clone();
                });
                emit_floating(&app, format!("识别暂不可用：{error}"), "error");
                sampler.note_error();
                wait_until(frame_started, sampler.error_backoff());
                continue;
            }
        };
        let capture_ms = capture_started.elapsed().as_secs_f64() * 1000.0;

        let is_first = previous_fine.is_none();
        let fine_changed = match &previous_fine {
            Some(previous) => {
                capture::changed_cells(previous, &captured.fine, capture::FINE_CELL_THRESHOLD)
            }
            None => (0..captured.fine.len()).collect(),
        };
        let coarse_changed = match &previous_coarse {
            Some(previous) => {
                capture::changed_cells(previous, &captured.coarse, capture::COARSE_CELL_THRESHOLD)
            }
            None => (0..captured.coarse.len()).collect(),
        };
        let bbox = change_bbox(
            &fine_changed,
            capture::FINE_COLS,
            capture::FINE_ROWS,
            captured.thumb_width,
            captured.thumb_height,
        );
        let area = bbox
            .map(|b| area_fraction(&b, captured.thumb_width, captured.thumb_height))
            .unwrap_or(0.0);
        previous_coarse = Some(captured.coarse.clone());
        previous_fine = Some(captured.fine.clone());

        if fine_changed.is_empty() && !is_first {
            count_round(&app);
            count_skipped(&app);
            sampler.note_idle();
            wait_until(frame_started, sampler.next_interval(Motion::Idle, 0));
            continue;
        }

        let motion = if coarse_changed.len() >= sampling::HIGH_MOTION_CELLS {
            Motion::HighActivity
        } else if area >= SIGNIFICANT_AREA_FRACTION || is_first {
            Motion::Significant
        } else {
            Motion::Small
        };

        let round_time = Instant::now();
        let mut send = false;
        let mut crop_rect: Option<(u32, u32, u32, u32)> = None;

        match motion {
            Motion::HighActivity => {
                // Video / scroll / animation: look less often but do not stop
                // looking entirely, so a late popup or toast is still caught.
                sampler.note_dynamic();
                let scene_confirmed = sampler.is_dynamic_scene();
                let gap_ok = round_time.duration_since(last_request_at) >= MIN_MODEL_GAP;
                if !scene_confirmed || gap_ok {
                    send = true;
                }
            }
            Motion::Significant => {
                sampler.note_change();
                send = true;
            }
            Motion::Small => {
                // Cursor blink / typing caret: confirm over a few frames and
                // suppress that low-value chatter afterwards.
                if let Some(candidate) = small_candidate.as_mut() {
                    if same_cell_set(&candidate.cells, &fine_changed) {
                        candidate.frames = candidate.frames.saturating_add(1);
                        candidate.seen_at = round_time;
                    } else {
                        *candidate = SmallCandidate::new(fine_changed.clone(), round_time);
                    }
                } else {
                    small_candidate = Some(SmallCandidate::new(fine_changed.clone(), round_time));
                }
                let candidate = small_candidate.as_ref().expect("candidate just built");
                let confirmed = candidate.frames >= 2;
                let stable = candidate.frames >= 4;
                let gap_ok = round_time.duration_since(last_request_at) >= MIN_MODEL_GAP;
                if confirmed && !stable && gap_ok && round_time >= low_info_suppress_until {
                    send = true;
                } else {
                    sampler.note_idle();
                }
            }
            Motion::Idle => {}
        }

        // Localised change → high-density crop; global change → full screen.
        if send && !is_first && area <= CROP_MAX_AREA_FRACTION {
            crop_rect = bbox;
        }

        if !send {
            count_round(&app);
            count_skipped(&app);
            wait_until(
                frame_started,
                sampler.next_interval(motion, coarse_changed.len()),
            );
            continue;
        }

        last_request_at = round_time;
        count_round(&app);
        count_request(&app);

        let encode_started = Instant::now();
        let input = match captured.build_input(crop_rect, multi_image) {
            Ok(input) => input,
            Err(error) => {
                emit_floating(&app, format!("识别暂不可用：{error}"), "error");
                wait_until(
                    frame_started,
                    sampler.next_interval(motion, coarse_changed.len()),
                );
                continue;
            }
        };
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1000.0;

        let app_for_delta = app.clone();
        let task_for_delta = task_id.clone();
        let result = recognize(&endpoint, &goal, &input, move |text| {
            if task_active(&app_for_delta, &task_for_delta) {
                emit_floating(&app_for_delta, text, "recognizing");
            }
        });

        let total_ms = frame_started.elapsed().as_secs_f64() * 1000.0;
        match result {
            Ok(result) => {
                sampler.note_success();
                let readability = textsim::classify_readability(&result.text);
                let low_info = textsim::is_low_information(&result.text);

                let contradicted = !last_text.is_empty()
                    && round_time >= contradiction_suppress_until
                    && textsim::contradicts(&last_text, &result.text);
                if contradicted {
                    // Consecutive results disagree: be conservative instead of
                    // printing an apparently-firm but unstable fact.
                    contradiction_suppress_until = round_time + CONTRADICTION_COOLDOWN;
                    let conservative = "画面内容存在变化，但关键文字暂无法稳定确认。";
                    emit_floating(&app, conservative, "recognizing");
                    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                        snapshot.model_ready = true;
                        snapshot.message = conservative.into();
                    });
                    persist_latest_result(&task_id, conservative);
                    last_text = result.text;
                } else if textsim::should_refresh(&last_text, &result.text) {
                    let timestamp = now();
                    update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                        snapshot.model_ready = true;
                        snapshot.message = result.text.clone();
                    });
                    persist_latest_result(&task_id, &result.text);
                    emit_recognition(
                        &app,
                        RecognitionEvent {
                            task_id: task_id.clone(),
                            phase: "recognizing".into(),
                            title: "屏幕识别结果".into(),
                            detail: result.text.clone(),
                            timestamp,
                            requires_confirmation: false,
                            complete: false,
                            ok: true,
                        },
                    );
                    emit_floating(&app, result.text.clone(), "recognizing");
                    last_text = result.text;
                }

                if low_info {
                    low_info_suppress_until = round_time + LOW_INFO_SUPPRESS;
                } else {
                    low_info_suppress_until = round_time;
                }
                record_metrics(
                    &app,
                    OpsMetrics {
                        capture_ms: Some(capture_ms),
                        encode_ms: Some(encode_ms),
                        first_token_ms: result.first_token_ms,
                        generate_ms: result.generate_ms,
                        total_ms: Some(total_ms),
                        prompt_tokens: result.prompt_tokens,
                        completion_tokens: result.completion_tokens,
                        readability: Some(readability_label(readability).into()),
                        input_kind: Some(if input.full_screen() {
                            "full".into()
                        } else {
                            "crop".into()
                        }),
                        server: None,
                        error: None,
                    },
                );
            }
            Err(error) => {
                // Service / parse errors are clearly separated from visual
                // uncertainty by the message itself and by the error metric.
                sampler.note_error();
                update_snapshot(&app.state::<RuntimeState>(), |snapshot| {
                    snapshot.message = error.clone();
                });
                emit_floating(&app, format!("识别暂不可用：{error}"), "error");
                record_metrics(
                    &app,
                    OpsMetrics {
                        capture_ms: Some(capture_ms),
                        encode_ms: Some(encode_ms),
                        first_token_ms: None,
                        generate_ms: None,
                        total_ms: Some(total_ms),
                        prompt_tokens: None,
                        completion_tokens: None,
                        readability: None,
                        input_kind: None,
                        server: None,
                        error: Some(error),
                    },
                );
            }
        }

        wait_until(
            frame_started,
            sampler.next_interval(motion, coarse_changed.len()),
        );
    }
}

#[tauri::command]
fn stop_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    let snapshot = update_snapshot(&state, |s| {
        s.phase = "stopped".into();
        s.message = "好，我先不看了。".into();
    });
    emit_floating(&app, "好，我先歇一会儿。", "stopped");
    let _ = hide_floating_window(app);
    snapshot
}

#[tauri::command]
fn pause_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    stop_runtime(app, state)
}

fn ensure_floating_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    if let Some(window) = app.get_webview_window(FLOATING_LABEL) {
        apply_floating_size_limits(&window);
        return Ok(window);
    }

    // Fallback when the preconfigured window is missing (e.g. older builds).
    // Load with an explicit query so the frontend can still detect floating mode.
    let window = WebviewWindowBuilder::new(
        app,
        FLOATING_LABEL,
        WebviewUrl::App("index.html?window=floating".into()),
    )
    .title("baodou")
    .inner_size(FLOATING_WIDTH, FLOATING_HEIGHT)
    .min_inner_size(FLOATING_MIN_WIDTH, FLOATING_MIN_HEIGHT)
    .max_inner_size(FLOATING_MAX_WIDTH, FLOATING_MAX_HEIGHT)
    .visible(false)
    .always_on_top(true)
    .decorations(false)
    .transparent(true)
    .shadow(false)
    .skip_taskbar(true)
    .resizable(false)
    .focused(false)
    .build()
    .map_err(|e| format!("创建悬浮窗失败：{e}"))?;
    apply_floating_size_limits(&window);
    Ok(window)
}

fn apply_floating_size_limits(window: &WebviewWindow) {
    let _ = window.set_min_size(Some(tauri::Size::Logical(tauri::LogicalSize::new(
        FLOATING_MIN_WIDTH,
        FLOATING_MIN_HEIGHT,
    ))));
    let _ = window.set_max_size(Some(tauri::Size::Logical(tauri::LogicalSize::new(
        FLOATING_MAX_WIDTH,
        FLOATING_MAX_HEIGHT,
    ))));
}

fn position_floating_window(
    app: &AppHandle,
    window: &WebviewWindow,
    width: f64,
    height: f64,
) -> Result<(), String> {
    let monitor = app
        .primary_monitor()
        .map_err(|e| e.to_string())?
        .or_else(|| {
            app.available_monitors()
                .ok()
                .and_then(|list| list.into_iter().next())
        })
        .ok_or_else(|| "无法获取显示器信息".to_string())?;

    let scale = monitor.scale_factor();
    let screen = monitor.size();
    let work_w = screen.width as f64 / scale;
    let work_h = screen.height as f64 / scale;
    let x = (work_w - width - 24.0).max(12.0);
    let y = (work_h - height - 56.0).max(12.0);

    window
        .set_size(tauri::Size::Logical(tauri::LogicalSize::new(width, height)))
        .map_err(|e| format!("设置悬浮窗尺寸失败：{e}"))?;
    window
        .set_position(Position::Logical(LogicalPosition::new(x, y)))
        .map_err(|e| format!("定位悬浮窗失败：{e}"))?;
    Ok(())
}

#[tauri::command]
fn show_floating_window(app: AppHandle) -> Result<(), String> {
    let window = ensure_floating_window(&app)?;
    position_floating_window(&app, &window, FLOATING_WIDTH, FLOATING_HEIGHT)?;
    window
        .set_always_on_top(true)
        .map_err(|e| format!("置顶悬浮窗失败：{e}"))?;
    window.show().map_err(|e| format!("显示悬浮窗失败：{e}"))?;
    // Keep the overlay above other windows without aggressively stealing keyboard focus.
    let _ = window.unminimize();
    Ok(())
}

#[tauri::command]
fn resize_floating_window(app: AppHandle, width: f64, height: f64) -> Result<(), String> {
    let Some(window) = app.get_webview_window(FLOATING_LABEL) else {
        return Ok(());
    };
    let width = width.round().clamp(FLOATING_MIN_WIDTH, FLOATING_MAX_WIDTH);
    let height = height.round().clamp(FLOATING_MIN_HEIGHT, FLOATING_MAX_HEIGHT);
    let scale = window.scale_factor().unwrap_or(1.0);
    let current_size = window.inner_size().map_err(|e| e.to_string())?;
    let current_pos = window.outer_position().map_err(|e| e.to_string())?;
    let current_w = f64::from(current_size.width) / scale;
    let current_h = f64::from(current_size.height) / scale;
    let current_x = f64::from(current_pos.x) / scale;
    let current_y = f64::from(current_pos.y) / scale;

    if (current_w - width).abs() < 0.5 && (current_h - height).abs() < 0.5 {
        return Ok(());
    }

    let mut x = current_x + current_w - width;
    let mut y = current_y + current_h - height;
    if let Ok(Some(monitor)) = window.current_monitor() {
        let monitor_scale = monitor.scale_factor();
        let screen = monitor.size();
        let work_w = f64::from(screen.width) / monitor_scale;
        let work_h = f64::from(screen.height) / monitor_scale;
        x = x.clamp(12.0, (work_w - width - 12.0).max(12.0));
        y = y.clamp(12.0, (work_h - height - 12.0).max(12.0));
    }

    let size = tauri::Size::Logical(tauri::LogicalSize::new(width, height));
    let position = Position::Logical(LogicalPosition::new(x, y));
    // Expand left/up first so the pet stays put; shrink the HWND before moving.
    let expanding = width > current_w + 0.5 || height > current_h + 0.5;
    if expanding {
        window
            .set_position(position)
            .map_err(|e| format!("定位悬浮窗失败：{e}"))?;
        window
            .set_size(size)
            .map_err(|e| format!("设置悬浮窗尺寸失败：{e}"))?;
    } else {
        window
            .set_size(size)
            .map_err(|e| format!("设置悬浮窗尺寸失败：{e}"))?;
        window
            .set_position(position)
            .map_err(|e| format!("定位悬浮窗失败：{e}"))?;
    }
    Ok(())
}

#[tauri::command]
fn hide_floating_window(app: AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window(FLOATING_LABEL) {
        window.hide().map_err(|e| format!("隐藏悬浮窗失败：{e}"))?;
    }
    Ok(())
}

fn show_main_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.unminimize();
        let _ = window.set_focus();
    }
}

fn build_tray(app: &AppHandle) -> tauri::Result<()> {
    let menu = Menu::new(app)?;
    let show = MenuItem::with_id(app, TRAY_SHOW_ID, "显示主界面", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, TRAY_EXIT_ID, "退出", true, None::<&str>)?;
    menu.append_items(&[&show, &quit])?;

    TrayIconBuilder::with_id("baodou-tray")
        .icon(
            app.default_window_icon()
                .cloned()
                .expect("missing application icon"),
        )
        .tooltip("baodou")
        .menu(&menu)
        .on_menu_event(|app, event| match event.id().as_ref() {
            TRAY_SHOW_ID => show_main_window(app),
            TRAY_EXIT_ID => {
                stop_model(app);
                app.exit(0);
            }
            _ => {}
        })
        .build(app)?;
    Ok(())
}

pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeState::default())
        .manage(ModelState::default())
        .setup(|app| {
            let _ = data_dir().map_err(std::io::Error::other)?;
            let _ = database().map_err(std::io::Error::other)?;
            let _ = persist_config_file(&load_model_config());
            build_tray(app.handle())?;

            if let Some(main) = app.get_webview_window("main") {
                let main_for_close = main.clone();
                main.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let _ = main_for_close.hide();
                    }
                });
            }

            // Ensure the floating webview is created at startup (hidden),
            // so show/hide later only toggles visibility.
            if let Ok(window) = ensure_floating_window(app.handle()) {
                let _ = position_floating_window(
                    app.handle(),
                    &window,
                    FLOATING_WIDTH,
                    FLOATING_HEIGHT,
                );
                let _ = window.hide();
            }

            let handle = app.handle().clone();
            std::thread::spawn(move || supervise_model(handle));

            let handle = app.handle().clone();
            std::thread::spawn(move || poll_server_metrics(handle));
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
            hide_floating_window,
            resize_floating_window
        ])
        .run(tauri::generate_context!())
        .expect("error while running baodou desktop");
}
