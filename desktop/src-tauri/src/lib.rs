use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

const PROTOCOL_VERSION: &str = "1.0.0";

#[derive(Default)]
struct RuntimeState(Mutex<RuntimeSnapshot>);

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
            mode: "mock · dry-run".into(),
            phase: "idle".into(),
            connected: true,
            inference_backend: "llama.cpp SYCL / mock".into(),
            device: "SYCL0 · Intel Arc".into(),
            model_ready: true,
            task_id: None,
            goal: None,
            message: "本地运行时已就绪".into(),
        }
    }
}

#[derive(Clone, Deserialize)]
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
    raw: Option<serde_json::Value>,
}

fn emit(app: &AppHandle, event: TaskEvent) {
    let _ = app.emit("task-event", event);
}

fn set_snapshot(state: &State<'_, RuntimeState>, update: impl FnOnce(&mut RuntimeSnapshot)) -> RuntimeSnapshot {
    let mut snapshot = state.0.lock().expect("runtime state poisoned");
    update(&mut snapshot);
    snapshot.clone()
}

#[tauri::command]
fn get_runtime(state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    state.0.lock().expect("runtime state poisoned").clone()
}

#[tauri::command]
fn pause_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    let snapshot = set_snapshot(&state, |s| {
        s.phase = "paused".into();
        s.message = "已暂停，等待你的继续指令".into();
    });
    emit(&app, TaskEvent { task_id: snapshot.task_id.clone().unwrap_or_default(), phase: "paused".into(), title: "任务已暂停".into(), detail: snapshot.message.clone(), timestamp: now(), requires_confirmation: false, complete: false, ok: true, raw: None });
    snapshot
}

#[tauri::command]
fn stop_runtime(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    let snapshot = set_snapshot(&state, |s| {
        s.phase = "stopped".into();
        s.message = "已停止当前任务，未继续发送操作".into();
    });
    emit(&app, TaskEvent { task_id: snapshot.task_id.clone().unwrap_or_default(), phase: "stopped".into(), title: "任务已停止".into(), detail: snapshot.message.clone(), timestamp: now(), requires_confirmation: false, complete: true, ok: true, raw: None });
    snapshot
}

#[tauri::command]
fn run_task(app: AppHandle, state: State<'_, RuntimeState>, request: TaskRequest) -> String {
    let task_id = Uuid::new_v4().to_string();
    let mode = if request.live { "live · confirmation gate" } else { "mock · dry-run" };
    let snapshot = set_snapshot(&state, |s| {
        s.mode = mode.into();
        s.phase = "observing".into();
        s.task_id = Some(task_id.clone());
        s.goal = Some(request.goal.clone());
        s.message = "正在采集屏幕并建立 UI 元素索引".into();
    });
    emit(&app, TaskEvent { task_id: task_id.clone(), phase: "observing".into(), title: "正在观察桌面".into(), detail: snapshot.message.clone(), timestamp: now(), requires_confirmation: false, complete: false, ok: true, raw: None });

    let app_clone = app.clone();
    let event_task_id = task_id.clone();
    std::thread::spawn(move || {
        let command = if request.live && request.auto_confirm { "--live" } else { "--mock" };
        let mut cmd = Command::new("conda");
        cmd.args(["run", "--no-capture-output", "-n", "dev", "python", "-m", "frontend.cli", "ui", "run", "--goal", &request.goal, command, "--preview-only"])
            .stdout(Stdio::piped()).stderr(Stdio::piped());
        let output = cmd.output();
        let (phase, title, detail, ok, raw) = match output {
            Ok(result) => {
                let text = String::from_utf8_lossy(&result.stdout).to_string();
                let parsed = text.lines().rev().find_map(|line| serde_json::from_str::<serde_json::Value>(line).ok());
                ("awaiting_user", "计划已准备".into(), parsed.as_ref().and_then(|v| v.get("observation")).and_then(|v| v.as_str()).unwrap_or("已完成屏幕观察和低风险计划预览").into(), result.status.success(), parsed)
            }
            Err(error) => ("error", "运行时连接失败".into(), format!("无法启动 conda dev：{error}"), false, None),
        };
        emit(&app_clone, TaskEvent { task_id: event_task_id, phase: phase.into(), title, detail, timestamp: now(), requires_confirmation: phase == "awaiting_user", complete: false, ok, raw });
    });
    task_id
}

#[tauri::command]
fn confirm_task(app: AppHandle, state: State<'_, RuntimeState>) -> RuntimeSnapshot {
    let snapshot = set_snapshot(&state, |s| {
        s.phase = "executing".into();
        s.message = "已确认，正在交给受控 agent 执行".into();
    });
    emit(&app, TaskEvent { task_id: snapshot.task_id.clone().unwrap_or_default(), phase: "executing".into(), title: "已确认执行".into(), detail: snapshot.message.clone(), timestamp: now(), requires_confirmation: false, complete: false, ok: true, raw: None });
    let app_clone = app.clone();
    let task_id = snapshot.task_id.clone().unwrap_or_default();
    let goal = snapshot.goal.clone().unwrap_or_default();
    let live = snapshot.mode.starts_with("live");
    std::thread::spawn(move || {
        let mut cmd = Command::new("conda");
        let args: Vec<&str> = if live {
            vec!["run", "--no-capture-output", "-n", "dev", "python", "-m", "frontend.cli", "agent", "run", "--goal", &goal, "--live", "--yes", "--actuator", "win"]
        } else {
            vec!["run", "--no-capture-output", "-n", "dev", "python", "-m", "frontend.cli", "agent", "run", "--goal", &goal, "--mock", "--yes"]
        };
        let result = cmd.args(args).stdout(Stdio::piped()).stderr(Stdio::piped()).output();
        let (phase, title, detail, ok, raw) = match result {
            Ok(output) => {
                let text = String::from_utf8_lossy(&output.stdout).to_string();
                let parsed = text.lines().rev().find_map(|line| serde_json::from_str::<serde_json::Value>(line).ok());
                let state = parsed.as_ref().and_then(|v| v.get("task_state")).and_then(|v| v.as_str()).unwrap_or("completed");
                (if output.status.success() { "completed" } else { "paused" }, if output.status.success() { "任务已完成" } else { "任务已暂停" }, format!("agent 状态：{state}"), output.status.success(), parsed)
            }
            Err(error) => ("error", "执行器启动失败", error.to_string(), false, None),
        };
        emit(&app_clone, TaskEvent { task_id, phase: phase.into(), title: title.into(), detail, timestamp: now(), requires_confirmation: false, complete: true, ok, raw });
    });
    snapshot
}

fn now() -> String {
    format!("{}Z", chrono_like_timestamp())
}

fn chrono_like_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs().to_string()).unwrap_or_else(|_| "0".into())
}

pub fn run() {
    tauri::Builder::default()
        .manage(RuntimeState::default())
        .invoke_handler(tauri::generate_handler![get_runtime, run_task, confirm_task, pause_runtime, stop_runtime])
        .run(tauri::generate_context!())
        .expect("error while running baodou desktop");
}
