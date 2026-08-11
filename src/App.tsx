import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { bridge } from "./bridge";
import type { ModelConfig, RuntimeSnapshot, TaskEvent } from "./types";
import {
  Activity, AlertTriangle, ArrowUpRight, Check, ChevronDown, CircleStop, Cpu,
  Eye, FileText, History, Keyboard, Layers3, LockKeyhole, Maximize2, Monitor,
  Minimize2, Pause, Plus, ScanSearch, Send, Settings, ShieldCheck, Sparkles, SquareTerminal,
  Timer, TriangleAlert, X,
} from "lucide-react";

const initialRuntime: RuntimeSnapshot = {
  protocolVersion: "1.0.0", mode: "native live · confirmation gate", phase: "idle", connected: true,
  inferenceBackend: "Rust planner / llama.cpp optional", device: "SYCL0 · Intel Arc", modelReady: false,
  taskId: null, goal: null, message: "Rust 本地运行时已就绪；不依赖 Python",
};

const suggestions = ["描述当前屏幕", "定位搜索按钮", "在当前输入框输入一段文本"];

function App() {
  const [runtime, setRuntime] = useState(initialRuntime);
  const [goal, setGoal] = useState("");
  const [live, setLive] = useState(true);
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [busy, setBusy] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [activePage, setActivePage] = useState<"session" | "history" | "logs" | "settings">("session");
  const [error, setError] = useState("");
  const [modelPath, setModelPath] = useState("");
  const [modelPathDraft, setModelPathDraft] = useState("");
  const [modelFilePathDraft, setModelFilePathDraft] = useState("");
  const [mmprojPathDraft, setMmprojPathDraft] = useState("");
  const [llamaUrlDraft, setLlamaUrlDraft] = useState("");
  const [modelConfigMessage, setModelConfigMessage] = useState("");

  useEffect(() => {
    bridge.runtime().then(setRuntime).catch(() => setError("无法连接本地运行时"));
    bridge.modelConfig().then((config) => { setModelPath(config.serverPath); setModelPathDraft(config.serverPath); setModelFilePathDraft(config.modelPath); setMmprojPathDraft(config.mmprojPath); setLlamaUrlDraft(config.llamaUrl); }).catch(() => setModelConfigMessage("无法读取模型配置"));
    const refreshRuntime = () => bridge.runtime().then(setRuntime).catch(() => undefined);
    const runtimeTimer = window.setInterval(refreshRuntime, 500);
    let cleanup: (() => void) | undefined;
    bridge.onTask((event) => {
      setEvents((current) => [...current.slice(-7), event]);
      setRuntime((current) => ({ ...current, phase: event.phase, message: event.detail }));
      if (event.complete || event.phase === "error") setBusy(false);
    }).then((unlisten) => { cleanup = unlisten; });
    return () => { window.clearInterval(runtimeTimer); cleanup?.(); };
  }, []);

  const latest = events.at(-1);
  const phaseLabel = useMemo(() => ({
    idle: "待命", observing: "观察屏幕", planning: "生成计划", awaiting_user: "等待确认",
    executing: "执行中", paused: "已暂停", stopped: "已停止", completed: "已完成", error: "需要处理",
  }[runtime.phase] ?? runtime.phase), [runtime.phase]);

  async function startTask() {
    const value = goal.trim();
    if (!value || busy) return;
    setError(""); setBusy(true); setEvents([]);
    setRuntime((current) => ({ ...current, phase: "observing", goal: value, message: "任务已提交，正在观察屏幕…" }));
    setEvents([{
      taskId: "local-submit", phase: "observing", title: "任务已提交",
      detail: "正在连接本地模型并采集屏幕，请稍候…", timestamp: new Date().toISOString(),
      requiresConfirmation: false, complete: false, ok: true,
    }]);
    try { await bridge.start(value, live, true); }
    catch (cause) { setBusy(false); setError(String(cause)); }
  }

  async function action(fn: () => Promise<RuntimeSnapshot>) {
    try { setRuntime(await fn()); } catch (cause) { setError(String(cause)); }
  }

  async function saveModelPath() {
    setModelConfigMessage("");
    try {
      const config: ModelConfig = await bridge.saveModelConfig({ serverPath: modelPathDraft.trim(), modelPath: modelFilePathDraft.trim(), mmprojPath: mmprojPathDraft.trim(), llamaUrl: llamaUrlDraft.trim() });
      setModelPath(config.serverPath);
      setModelPathDraft(config.serverPath);
      setModelFilePathDraft(config.modelPath);
      setMmprojPathDraft(config.mmprojPath);
      setLlamaUrlDraft(config.llamaUrl);
      setModelConfigMessage("路径已保存，重启应用后自动使用");
    } catch (cause) { setModelConfigMessage(String(cause)); }
  }

  const currentWindow = getCurrentWindow();
  const minimizeWindow = () => { void currentWindow.minimize(); };
  const toggleMaximizeWindow = () => { void currentWindow.toggleMaximize(); };
  const closeWindow = () => { void currentWindow.close(); };

  const canConfirm = runtime.phase === "awaiting_user";
  function dragWindow(event: React.MouseEvent<HTMLElement>) {
    // In a frameless Tauri window, explicitly delegate dragging to the native
    // shell. Interactive descendants must keep their normal pointer behavior.
    if (event.button !== 0) return;
    const target = event.target as HTMLElement;
    if (target.closest("button, input, textarea, select, a, [data-no-window-drag]")) return;
    void getCurrentWindow().startDragging().catch((cause) => {
      console.error("无法拖动窗口", cause);
    });
  }
  return (
    <main className="app-shell">
      <header className="topbar" data-tauri-drag-region onMouseDown={dragWindow}>
        <div className="brand"><div className="brand-mark"><Sparkles size={16} /></div><span>baodou</span><small>DESKTOP</small></div>
        <div className="topbar-center"><span className="traffic-dot" /> {runtime.mode}<span className="slash">/</span><span className="muted">本地运行</span></div>
        <div className="window-tools"><button className="icon-button" title="停止当前任务" onClick={() => action(bridge.stop)}><CircleStop size={16} /></button><button className="window-button" title="最小化" onClick={minimizeWindow}><Minimize2 size={15} /></button><button className="window-button" title="最大化 / 还原" onClick={toggleMaximizeWindow}><Maximize2 size={15} /></button><button className="window-button close-button" title="关闭" onClick={closeWindow}><X size={15} /></button></div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <button className="new-task" onClick={() => { setActivePage("session"); setShowDiagnostics(false); setGoal(""); setEvents([]); setRuntime({ ...initialRuntime }); }}><Plus size={17} /> 新任务 <span>⌘ K</span></button>
          <nav className="nav-group"><p>工作区</p><button className={`nav-item ${activePage === "session" ? "active" : ""}`} onClick={() => { setActivePage("session"); setShowDiagnostics(false); }}><Activity size={16} /> 当前会话 <em>{events.length || ""}</em></button><button className={`nav-item ${activePage === "history" ? "active" : ""}`} onClick={() => { setActivePage("history"); setShowDiagnostics(false); }}><History size={16} /> 任务历史</button><button className={`nav-item ${activePage === "logs" ? "active" : ""}`} onClick={() => { setActivePage("logs"); setShowDiagnostics(false); }}><FileText size={16} /> 运行日志</button><button className={`nav-item ${activePage === "settings" ? "active" : ""}`} onClick={() => { setActivePage("settings"); setShowDiagnostics(false); }}><Settings size={16} /> 设置</button></nav>
          <div className="sidebar-bottom"><div className="privacy-line"><LockKeyhole size={14} /><span>本地隐私模式</span><i /></div><div className="runtime-mini"><div className="mini-icon"><Cpu size={14} /></div><div><strong>Qwen3.5 · 2B</strong><small>{runtime.device}</small></div><ChevronDown size={14} /></div></div>
        </aside>

        <section className={`main-pane ${activePage !== "session" ? "page-active" : ""}`}>
          {activePage === "settings" && <SettingsPage modelPathDraft={modelPathDraft} setModelPathDraft={setModelPathDraft} modelFilePathDraft={modelFilePathDraft} setModelFilePathDraft={setModelFilePathDraft} mmprojPathDraft={mmprojPathDraft} setMmprojPathDraft={setMmprojPathDraft} llamaUrlDraft={llamaUrlDraft} setLlamaUrlDraft={setLlamaUrlDraft} saveModelPath={saveModelPath} modelConfigMessage={modelConfigMessage} />}
          {activePage === "history" && <SimplePage icon={<History size={22} />} title="任务历史" detail="任务历史记录将在本地任务存储接入后显示。" />}
          {activePage === "logs" && <SimplePage icon={<FileText size={22} />} title="运行日志" detail="运行日志会在模型启动和任务执行后显示。" />}
          <div className="content-scroll">
            <div className="eyebrow"><span className={`status-pip ${busy ? "live" : ""}`} /> SESSION / {runtime.phase.toUpperCase()}</div>
            <div className="intro"><h1>让电脑替你完成<br /><span>下一步。</span></h1><p>描述一个目标，baodou 会观察屏幕、整理计划，并在每次操作前交还控制权。</p></div>
            <div className="task-composer">
              <div className="composer-top"><span className="composer-label"><Sparkles size={15} /> 任务指令</span><span className="composer-hint">Enter 发送 · Shift Enter 换行</span></div>
              <textarea value={goal} onChange={(e) => setGoal(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); startTask(); } }} placeholder="例如：打开浏览器，搜索今天的天气" rows={3} />
              <div className="composer-footer"><div className="suggestions">{suggestions.map((item) => <button key={item} onClick={() => setGoal(item)}>{item}</button>)}</div><button className="send-button" onClick={startTask} disabled={!goal.trim() || busy}><Send size={16} /> {busy ? "运行中" : "开始"}</button></div>
            </div>
              <div className="mode-row"><button className={`mode-toggle ${live ? "selected" : ""}`} onClick={() => setLive(!live)}><span className="toggle"><i /></span><span><strong>{live ? "本地模型模式" : "原生预览模式"}</strong><small>{live ? "Rust 直连 llama-server；仍需逐步确认" : "Rust 截图与规则规划；不会注入键鼠事件"}</small></span></button><div className="safe-badge"><ShieldCheck size={15} /> 高风险操作自动拦截</div></div>

            <div className="section-heading"><div><span className="section-kicker">LIVE SESSION</span><h2>工作流</h2></div><span className={`phase-badge ${runtime.phase}`}>{phaseLabel}</span></div>
            <div className="timeline">{events.length === 0 ? <div className="empty-state"><ScanSearch size={21} /><span>输入目标后，这里会显示观察、计划、确认与验证过程。</span></div> : events.map((event, index) => <div className={`timeline-item ${index === events.length - 1 ? "current" : ""}`} key={`${event.taskId}-${index}`}><div className="timeline-line"><span className="timeline-icon">{event.phase === "awaiting_user" ? <TriangleAlert size={14} /> : event.phase === "completed" ? <Check size={14} /> : <Activity size={14} />}</span>{index < events.length - 1 && <i />}</div><div className="timeline-content"><div className="timeline-title"><strong>{event.title}</strong><time><Timer size={12} />刚刚</time></div><p>{event.detail}</p>{event.phase === "awaiting_user" && <div className="approval-inline"><button className="approve" onClick={() => action(bridge.confirm)}><Check size={14} /> 确认这一步</button><button className="reject" onClick={() => action(bridge.pause)}><X size={14} /> 暂停</button></div>}</div></div>)}</div>
          </div>
          <footer className="main-footer"><span><Keyboard size={14} /> 全局停止 <kbd>Ctrl</kbd><kbd>Alt</kbd><kbd>Esc</kbd></span><span className="footer-right"><span className="connection"><i /> Rust host connected</span><button onClick={() => setShowDiagnostics(!showDiagnostics)}>查看运行信息 <ArrowUpRight size={13} /></button></span></footer>
        </section>

        <aside className={`inspector ${showDiagnostics ? "open" : ""}`}><div className="inspector-title"><div><span className="section-kicker">RUNTIME</span><h3>运行状态</h3></div><button className="icon-button" onClick={() => setShowDiagnostics(false)}><X size={16} /></button></div><div className="screen-preview"><div className="preview-grid" /><div className="preview-window"><div /><div /><div /></div><span className="preview-label"><Eye size={12} /> 屏幕预览将在观察时出现</span></div><div className="inspector-section"><label>当前目标</label><p className="target-text">{runtime.goal || "尚未开始任务"}</p></div><div className="metric-list"><Metric icon={<Monitor size={15} />} name="屏幕采集" value={runtime.phase === "observing" ? "ACTIVE" : "READY"} /><Metric icon={<Cpu size={15} />} name="推理后端" value={runtime.inferenceBackend} /><Metric icon={<ScanSearch size={15} />} name="计划器" value="Rust planner" /><Metric icon={<Timer size={15} />} name="协议版本" value={runtime.protocolVersion} /></div><div className="model-settings"><label>llama-server.exe 路径</label><input value={modelPathDraft} onChange={(event) => setModelPathDraft(event.target.value)} /><label>模型 GGUF 路径</label><input value={modelFilePathDraft} onChange={(event) => setModelFilePathDraft(event.target.value)} /><label>MMPROJ 路径</label><input value={mmprojPathDraft} onChange={(event) => setMmprojPathDraft(event.target.value)} /><label>LLAMA URL</label><input value={llamaUrlDraft} onChange={(event) => setLlamaUrlDraft(event.target.value)} /><button onClick={saveModelPath}>保存模型配置</button>{modelConfigMessage && <small>{modelConfigMessage}</small>}</div><div className="inspector-note"><ShieldCheck size={17} /><div><strong>本地模型配置</strong><p>路径和 URL 保存在应用数据目录，重启应用后自动使用。</p></div></div>{error && <div className="error-box"><AlertTriangle size={15} /> {error}</div>}{canConfirm && <button className="confirm-large" onClick={() => action(bridge.confirm)}><Check size={16} /> 确认并继续</button>}{busy && <button className="pause-large" onClick={() => action(bridge.pause)}><Pause size={15} /> 暂停当前任务</button>}</aside>
      </div>
    </main>
  );
}

function SettingsPage(props: {
  modelPathDraft: string; setModelPathDraft: (value: string) => void;
  modelFilePathDraft: string; setModelFilePathDraft: (value: string) => void;
  mmprojPathDraft: string; setMmprojPathDraft: (value: string) => void;
  llamaUrlDraft: string; setLlamaUrlDraft: (value: string) => void;
  saveModelPath: () => void; modelConfigMessage: string;
}) {
  return <div className="settings-page"><div className="settings-heading"><span className="section-kicker">PREFERENCES</span><h1>设置</h1><p>配置本地模型运行时。修改后点击保存，下一次启动时自动使用。</p></div><div className="settings-form"><label>llama-server.exe 路径</label><input value={props.modelPathDraft} onChange={(event) => props.setModelPathDraft(event.target.value)} /><label>模型 GGUF 路径</label><input value={props.modelFilePathDraft} onChange={(event) => props.setModelFilePathDraft(event.target.value)} /><label>MMPROJ 路径</label><input value={props.mmprojPathDraft} onChange={(event) => props.setMmprojPathDraft(event.target.value)} /><label>LLAMA URL</label><input value={props.llamaUrlDraft} onChange={(event) => props.setLlamaUrlDraft(event.target.value)} /><button onClick={props.saveModelPath}>保存模型配置</button>{props.modelConfigMessage && <small>{props.modelConfigMessage}</small>}</div></div>;
}

function SimplePage({ icon, title, detail }: { icon: ReactNode; title: string; detail: string }) {
  return <div className="simple-page"><div className="simple-page-icon">{icon}</div><span className="section-kicker">WORKSPACE</span><h1>{title}</h1><p>{detail}</p></div>;
}

function Metric({ icon, name, value }: { icon: ReactNode; name: string; value: string }) { return <div className="metric"><span>{icon}</span><label>{name}</label><strong title={value}>{value}</strong></div>; }

export default App;
