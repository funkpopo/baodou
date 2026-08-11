import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { bridge } from "./bridge";
import type { ModelConfig, RuntimeSnapshot, TaskEvent } from "./types";
import {
  Activity, AlertTriangle, ArrowUp, Check, CircleStop, Cpu,
  Eye, FileText, History, Keyboard, LockKeyhole, Maximize2, MessageSquarePlus, Monitor,
  Minimize2, Pause, Plus, ScanSearch, Send, Settings, ShieldCheck, Sparkles,
  Timer, TriangleAlert, X,
} from "lucide-react";

const initialRuntime: RuntimeSnapshot = {
  protocolVersion: "1.0.0", mode: "native computer use", phase: "idle", connected: true,
  inferenceBackend: "Rust planner / llama.cpp optional", device: "SYCL0 · Intel Arc", modelReady: false,
  taskId: null, goal: null, message: "Rust 本地运行时已就绪；不依赖 Python",
};

const suggestions = ["在 VS Code 中提交当前代码", "打开浏览器搜索今天的天气", "切换到记事本并输入一段文本"];

type SessionTurn = {
  id: string;
  goal: string;
  result: string;
  phase: "completed" | "stopped" | "error";
};

function App() {
  const [runtime, setRuntime] = useState(initialRuntime);
  const [goal, setGoal] = useState("");
  const [events, setEvents] = useState<TaskEvent[]>([]);
  const [sessionTurns, setSessionTurns] = useState<SessionTurn[]>([]);
  const [busy, setBusy] = useState(false);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [conversationCollapsed, setConversationCollapsed] = useState(false);
  const [activePage, setActivePage] = useState<"session" | "history" | "logs" | "settings">("session");
  const [error, setError] = useState("");
  const [modelPathDraft, setModelPathDraft] = useState("");
  const [modelFilePathDraft, setModelFilePathDraft] = useState("");
  const [mmprojPathDraft, setMmprojPathDraft] = useState("");
  const [llamaUrlDraft, setLlamaUrlDraft] = useState("");
  const [modelConfigMessage, setModelConfigMessage] = useState("");
  const taskSubmissionLock = useRef(false);
  const archivedTaskIds = useRef(new Set<string>());
  const runtimeGoalRef = useRef<string | null>(null);
  const eventsRef = useRef<TaskEvent[]>([]);

  useEffect(() => {
    runtimeGoalRef.current = runtime.goal;
  }, [runtime.goal]);

  useEffect(() => {
    eventsRef.current = events;
  }, [events]);

  useEffect(() => {
    bridge.runtime().then(setRuntime).catch(() => setError("无法连接本地运行时"));
    bridge.modelConfig().then((config) => { setModelPathDraft(config.serverPath); setModelFilePathDraft(config.modelPath); setMmprojPathDraft(config.mmprojPath); setLlamaUrlDraft(config.llamaUrl); }).catch(() => setModelConfigMessage("无法读取模型配置"));
    const refreshRuntime = () => bridge.runtime().then(setRuntime).catch(() => undefined);
    const runtimeTimer = window.setInterval(refreshRuntime, 500);
    let cleanup: (() => void) | undefined;
    let cancelled = false;
    bridge.onTask((event) => {
      setEvents((current) => {
        const existingIndex = current.findIndex((item) => item.taskId === event.taskId && item.phase === event.phase && item.title === event.title);
        if (existingIndex >= 0) {
          const next = [...current];
          next[existingIndex] = event;
          eventsRef.current = next;
          return next;
        }
        const next = [...current.slice(-7), event];
        eventsRef.current = next;
        return next;
      });
      setRuntime((current) => {
        const next = { ...current, phase: event.phase, message: event.detail, taskId: event.taskId || current.taskId };
        if (event.phase === "observing" || event.phase === "planning") {
          // Keep ref aligned for terminal archival.
        }
        return next;
      });
      // The event is the source of truth for the task lifecycle. In particular,
      // completed/stopped are terminal states even when the command response
      // arrives before the event listener has flushed its React update.
      if (event.complete || ["completed", "stopped", "error"].includes(event.phase)) {
        setBusy(false);
        taskSubmissionLock.current = false;
        setConversationCollapsed(false);
        const archiveKey = `${event.taskId}:${event.phase}`;
        if (!archivedTaskIds.current.has(archiveKey) && event.phase !== "idle") {
          archivedTaskIds.current.add(archiveKey);
          const goalText = runtimeGoalRef.current?.trim() || "当前任务";
          const resultText = modelResult(eventsRef.current) || event.detail || "（无详细输出）";
          const phase = (event.phase === "stopped" || event.phase === "error" ? event.phase : "completed") as SessionTurn["phase"];
          setSessionTurns((turns) => {
            // Avoid duplicating the same terminal task id.
            if (turns.some((turn) => turn.id === archiveKey)) return turns;
            return [...turns, { id: archiveKey, goal: goalText, result: resultText, phase }].slice(-12);
          });
        }
      }
    }).then((unlisten) => { if (cancelled) unlisten(); else cleanup = unlisten; });
    return () => { cancelled = true; window.clearInterval(runtimeTimer); cleanup?.(); };
  }, []);

  const latest = events.at(-1);
  const phaseLabel = useMemo(() => ({
    idle: "待命", observing: "观察屏幕", planning: "生成计划", awaiting_user: "执行中",
    executing: "执行中", paused: "已暂停", stopped: "已停止", completed: "已完成", error: "需要处理",
  }[runtime.phase] ?? runtime.phase), [runtime.phase]);

  function resetSession() {
    setActivePage("session");
    setShowDiagnostics(false);
    setGoal("");
    setEvents([]);
    setSessionTurns([]);
    archivedTaskIds.current.clear();
    setRuntime({ ...initialRuntime });
    setBusy(false);
    taskSubmissionLock.current = false;
    setError("");
    setConversationCollapsed(false);
  }

  async function startTask(nextGoal?: string) {
    const value = (nextGoal ?? goal).trim();
    if (!value || busy || taskSubmissionLock.current) return;
    taskSubmissionLock.current = true;
    setError("");
    setBusy(true);
    setGoal(value);
    setActivePage("session");
    setConversationCollapsed(false);
    setEvents([]);
    runtimeGoalRef.current = value;
    setRuntime((current) => ({ ...current, phase: "observing", goal: value, message: "任务已提交，正在观察屏幕…" }));
    const bootstrap: TaskEvent = {
      taskId: "local-submit", phase: "observing", title: "任务已提交",
      detail: "正在连接本地模型并采集屏幕，请稍候…", timestamp: new Date().toISOString(),
      requiresConfirmation: false, complete: false, ok: true,
    };
    setEvents([bootstrap]);
    eventsRef.current = [bootstrap];
    try { await bridge.start(value); }
    catch (cause) { setBusy(false); taskSubmissionLock.current = false; setError(String(cause)); }
  }

  async function action(fn: () => Promise<RuntimeSnapshot>) {
    setError("");
    setRuntime((current) => ({ ...current, message: "正在处理请求…" }));
    try { setRuntime(await fn()); } catch (cause) { setError(String(cause)); setRuntime((current) => ({ ...current, message: "操作失败，请查看错误信息" })); }
  }

  async function saveModelPath() {
    setModelConfigMessage("");
    try {
      const config: ModelConfig = await bridge.saveModelConfig({ serverPath: modelPathDraft.trim(), modelPath: modelFilePathDraft.trim(), mmprojPath: mmprojPathDraft.trim(), llamaUrl: llamaUrlDraft.trim() });
      setModelPathDraft(config.serverPath);
      setModelFilePathDraft(config.modelPath);
      setMmprojPathDraft(config.mmprojPath);
      setLlamaUrlDraft(config.llamaUrl);
      setModelConfigMessage("配置已保存，模型正在重启");
    } catch (cause) { setModelConfigMessage(String(cause)); }
  }

  const currentWindow = getCurrentWindow();
  const minimizeWindow = () => { void currentWindow.minimize().catch((cause) => console.error("窗口最小化失败", cause)); };
  const toggleMaximizeWindow = () => { void currentWindow.toggleMaximize().catch((cause) => console.error("窗口最大化失败", cause)); };
  const closeWindow = () => { void currentWindow.close().catch((cause) => console.error("窗口关闭失败", cause)); };

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
        <div className="brand">
          <div className="brand-mark"><Sparkles size={15} /></div>
          <span>baodou</span>
          <small>DESKTOP</small>
        </div>
        <div className="topbar-center">
          <span className="traffic-dot" />
          {runtime.mode}
          <span className="slash">/</span>
          <span className="muted">本地运行</span>
        </div>
        <div className="window-tools">
          <button className="window-button" title="最小化" onClick={minimizeWindow}><Minimize2 size={14} /></button>
          <button className="window-button" title="最大化 / 还原" onClick={toggleMaximizeWindow}><Maximize2 size={14} /></button>
          <button className="window-button close-button" title="关闭" onClick={closeWindow}><X size={14} /></button>
        </div>
      </header>

      <div className="workspace">
        <aside className="sidebar">
          <button className="new-task pressable" onClick={resetSession}>
            <Plus size={16} /> 新任务 <span>⌘ K</span>
          </button>

          <nav className="nav-group">
            <p>工作区</p>
            <button
              className={`nav-item ${activePage === "session" ? "active" : ""}`}
              onClick={() => { setActivePage("session"); setShowDiagnostics(false); }}
            >
              <Activity size={15} /> 当前会话
              {(events.length > 0 || sessionTurns.length > 0) && <em>{sessionTurns.length + (events.length ? 1 : 0)}</em>}
            </button>
            <button
              className={`nav-item ${activePage === "history" ? "active" : ""}`}
              onClick={() => { setActivePage("history"); setShowDiagnostics(false); }}
            >
              <History size={15} /> 任务历史
            </button>
            <button
              className={`nav-item ${activePage === "logs" ? "active" : ""}`}
              onClick={() => { setActivePage("logs"); setShowDiagnostics(false); }}
            >
              <FileText size={15} /> 运行日志
            </button>
            <button
              className={`nav-item ${activePage === "settings" ? "active" : ""}`}
              onClick={() => { setActivePage("settings"); setShowDiagnostics(false); }}
            >
              <Settings size={15} /> 设置
            </button>
          </nav>

          <div className="sidebar-bottom">
            <div className="privacy-line">
              <LockKeyhole size={13} />
              <span>本地隐私模式</span>
              <i />
            </div>
            <div
              className={`runtime-mini ${runtime.modelReady ? "model-online" : "model-offline"}`}
              title={runtime.message}
            >
              <div className="mini-icon"><Cpu size={13} /></div>
              <div>
                <strong>{runtime.modelReady ? "模型已连接" : "模型未就绪"}</strong>
                <small>{runtime.inferenceBackend}</small>
              </div>
              <span className="model-status-dot" />
            </div>
          </div>
        </aside>

        <section className={`main-pane ${activePage !== "session" ? "page-active" : ""}`}>
          {activePage === "settings" && (
            <SettingsPage
              modelPathDraft={modelPathDraft}
              setModelPathDraft={setModelPathDraft}
              modelFilePathDraft={modelFilePathDraft}
              setModelFilePathDraft={setModelFilePathDraft}
              mmprojPathDraft={mmprojPathDraft}
              setMmprojPathDraft={setMmprojPathDraft}
              llamaUrlDraft={llamaUrlDraft}
              setLlamaUrlDraft={setLlamaUrlDraft}
              saveModelPath={saveModelPath}
              modelConfigMessage={modelConfigMessage}
            />
          )}
          {activePage === "history" && (
            <SimplePage icon={<History size={20} />} title="任务历史" detail="任务历史记录将在本地任务存储接入后显示。" />
          )}
          {activePage === "logs" && (
            <SimplePage icon={<FileText size={20} />} title="运行日志" detail="运行日志会在模型启动和任务执行后显示。" />
          )}

          <div className="content-scroll">
            <div className="eyebrow">
              <span className={`status-pip ${busy ? "live" : ""}`} />
              SESSION / {runtime.phase.toUpperCase()}
            </div>

            <div className={`intro ${events.length || sessionTurns.length ? "has-session" : ""}`}>
              <h1>
                {busy
                  ? "任务处理中"
                  : events.length || sessionTurns.length
                    ? "可以继续对话"
                    : <>让电脑替你完成<br /><span>下一步。</span></>}
              </h1>
              <p>
                {events.length || sessionTurns.length
                  ? (runtime.goal || sessionTurns.at(-1)?.goal || "当前会话")
                  : "描述最终目标，baodou 会持续观察、操作并验证，直到目标真正完成。"}
              </p>
            </div>

            {events.length === 0 && sessionTurns.length === 0 && (
              <div className="task-composer">
                <div className="composer-top">
                  <span className="composer-label"><Sparkles size={14} /> 任务指令</span>
                  <span className="composer-hint">Enter 发送 · Shift+Enter 换行</span>
                </div>
                <textarea
                  value={goal}
                  onChange={(e) => setGoal(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void startTask();
                    }
                  }}
                  placeholder="例如：打开浏览器，搜索今天的天气"
                  rows={3}
                />
                <div className="foreground-notice">
                  <Monitor size={12} />
                  <span>运行时会接管当前 Windows 前台窗口</span>
                  <kbd>Ctrl</kbd><kbd>Alt</kbd><kbd>Esc</kbd>
                  <span>随时中止</span>
                </div>
                <div className="composer-footer">
                  <div className="suggestions">
                    {suggestions.map((item) => (
                      <button key={item} className="pressable" onClick={() => setGoal(item)}>{item}</button>
                    ))}
                  </div>
                  <button
                    className="send-button pressable"
                    onClick={() => void startTask()}
                    disabled={!goal.trim() || busy}
                  >
                    <Send size={15} /> {busy ? "运行中" : "开始"}
                  </button>
                </div>
              </div>
            )}

            <BotStage
              phase={runtime.phase}
              busy={busy}
              status={latest?.title || phaseLabel}
              detail={runtime.message}
              stepCount={events.filter((event) => event.phase === "executing").length}
              result={latest?.phase === "completed" ? modelResult(events) : sessionTurns.at(-1)?.phase === "completed" ? sessionTurns.at(-1)?.result : undefined}
            />

            {/* Kept in DOM for structure; visually hidden via CSS — workflow lives in the right panel */}
            <div className="section-heading">
              <div>
                <span className="section-kicker">LIVE SESSION</span>
                <h2>工作流</h2>
              </div>
              <span className={`phase-badge ${runtime.phase}`}>{phaseLabel}</span>
            </div>
            <div className="timeline">
              {events.length === 0 ? (
                <div className="empty-state">
                  <ScanSearch size={20} />
                  <span>输入目标后，这里会显示观察、决定、操作与验证过程。</span>
                </div>
              ) : (
                events.map((event, index) => (
                  <div
                    className={`timeline-item ${index === events.length - 1 ? "current" : ""}`}
                    key={`${event.taskId}-${index}`}
                  >
                    <div className="timeline-line">
                      <span className="timeline-icon">
                        {event.phase === "awaiting_user" ? <TriangleAlert size={13} />
                          : event.phase === "completed" ? <Check size={13} />
                            : <Activity size={13} />}
                      </span>
                      {index < events.length - 1 && <i />}
                    </div>
                    <div className="timeline-content">
                      <div className="timeline-title">
                        <strong>{event.title}</strong>
                        <time><Timer size={11} />刚刚</time>
                      </div>
                      <p>{event.detail}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          <footer className="main-footer">
            <span><Keyboard size={12} /> 全局停止 <kbd>Ctrl</kbd><kbd>Alt</kbd><kbd>Esc</kbd></span>
            <span className="footer-right">
              <span className="connection"><i /> Rust host connected</span>
            </span>
          </footer>
        </section>

        <aside className={`workflow-sidebar ${activePage !== "session" ? "hidden" : ""} ${conversationCollapsed ? "collapsed" : ""}`}>
          <WorkflowPanel
            events={events}
            sessionTurns={sessionTurns}
            busy={busy}
            currentGoal={runtime.goal}
            onAction={action}
            onFollowUp={(value) => void startTask(value)}
            onNewSession={resetSession}
            collapsed={conversationCollapsed}
            onToggle={() => setConversationCollapsed(!conversationCollapsed)}
          />
        </aside>

        <aside className={`inspector ${showDiagnostics ? "open" : ""}`}>
          <div className="inspector-title">
            <div>
              <span className="section-kicker">RUNTIME</span>
              <h3>运行状态</h3>
            </div>
            <button className="icon-button" onClick={() => setShowDiagnostics(false)}><X size={15} /></button>
          </div>
          <div className="screen-preview">
            <div className="preview-grid" />
            <div className="preview-window"><div /><div /><div /></div>
            <span className="preview-label"><Eye size={11} /> 屏幕预览将在观察时出现</span>
          </div>
          <div className="inspector-section">
            <label>当前目标</label>
            <p className="target-text">{runtime.goal || "尚未开始任务"}</p>
          </div>
          <div className="metric-list">
            <Metric icon={<Monitor size={14} />} name="屏幕采集" value={runtime.phase === "observing" ? "ACTIVE" : "READY"} />
            <Metric icon={<Cpu size={14} />} name="推理后端" value={runtime.inferenceBackend} />
            <Metric icon={<ScanSearch size={14} />} name="计划器" value="Rust planner" />
            <Metric icon={<Timer size={14} />} name="协议版本" value={runtime.protocolVersion} />
          </div>
          <div className="model-settings">
            <label>llama-server.exe 路径</label>
            <input value={modelPathDraft} onChange={(event) => setModelPathDraft(event.target.value)} />
            <label>模型 GGUF 路径</label>
            <input value={modelFilePathDraft} onChange={(event) => setModelFilePathDraft(event.target.value)} />
            <label>MMPROJ 路径</label>
            <input value={mmprojPathDraft} onChange={(event) => setMmprojPathDraft(event.target.value)} />
            <label>LLAMA URL</label>
            <input value={llamaUrlDraft} onChange={(event) => setLlamaUrlDraft(event.target.value)} />
            <button className="pressable" onClick={saveModelPath}>保存模型配置</button>
            {modelConfigMessage && <small>{modelConfigMessage}</small>}
          </div>
          <div className="inspector-note">
            <ShieldCheck size={16} />
            <div>
              <strong>本地模型配置</strong>
              <p>路径和 URL 保存在应用数据目录，重启应用后自动使用。</p>
            </div>
          </div>
          {error && <div className="error-box"><AlertTriangle size={14} /> {error}</div>}
          {busy && (
            <button className="pause-large pressable" onClick={() => action(bridge.pause)}>
              <Pause size={14} /> 暂停当前任务
            </button>
          )}
        </aside>
      </div>
    </main>
  );
}

/** Strip presentation wrappers from model output. */
function normalizeModelDetail(detail: string): string {
  let text = detail.trim();
  for (const prefix of ["模型结果："]) {
    if (text.startsWith(prefix)) text = text.slice(prefix.length).trim();
  }
  const fenced = text.match(/^```(?:\w+)?\s*\r?\n?([\s\S]*?)(?:\r?\n?```\s*)?$/);
  if (fenced) text = fenced[1].trim();
  const embedded = text.match(/```(?:\w+)?\s*\r?\n?([\s\S]*?)\r?\n?```/);
  if (embedded) text = embedded[1].trim();
  return text;
}

function firstTaggedPlan(detail: string): string {
  const segments = detail.replace(/；/g, ";").split(/([;\r\n])/);
  let statusCount = 0;
  let result = "";
  for (const segment of segments) {
    const trimmed = segment.trimStart();
    if (/^(?:STATUS|状态)\s*[:：]/i.test(trimmed)) {
      statusCount += 1;
      if (statusCount > 1) break;
    }
    if (trimmed.startsWith("END_PLAN")) break;
    result += segment;
  }
  return result.trim();
}

function llmResult(detail: string, streaming = false): string {
  const normalized = firstTaggedPlan(normalizeModelDetail(detail));
  if (!normalized) return streaming ? "正在生成模型结果…" : "";

  const fields = new Map<string, string>();
  for (const line of normalized.replace(/[；;]/g, "\n").split(/\r?\n/)) {
    const pair = line.trim().match(/^([^:：]+)[:：]\s*(.*)$/);
    if (!pair) continue;
    const key = pair[1].trim().toUpperCase();
    if (!fields.has(key)) fields.set(key, pair[2].trim());
  }
  const value = (...keys: string[]) => keys.map((key) => fields.get(key)).find(Boolean) ?? "";
  const status = value("STATUS", "状态").toUpperCase();
  const observation = value("OBSERVATION", "观察");
  const action = value("ACTION", "动作");
  const target = value("TARGET", "目标");
  const expected = value("EXPECTED", "预期");
  const summary = value("SUMMARY", "总结", "结果");
  if (!status && !observation && !action) return normalized;
  if (status.includes("DONE") || status.includes("完成")) return summary || observation;
  const actionLine = [action, target].filter(Boolean).join(" · ");
  return [observation, actionLine ? `下一步：${actionLine}` : "", expected ? `预期：${expected}` : "", streaming ? "（正在生成计划…）" : ""]
    .filter(Boolean)
    .join("\n");
}

function modelOutputEvent(events: TaskEvent[]) {
  const latest = events.at(-1);
  if (latest && ["completed", "stopped", "error"].includes(latest.phase)) return latest;
  return [...events].reverse().find((event) => event.phase === "planning" && event.title === "模型输出中");
}

function modelResult(events: TaskEvent[]) {
  const source = modelOutputEvent(events);
  if (!source) return "";
  const latest = events.at(-1);
  const streaming = source.phase === "planning" && latest?.phase !== "completed" && latest?.phase !== "stopped" && latest?.phase !== "error";
  return llmResult(source.detail, streaming);
}

function WorkflowPanel({
  events,
  sessionTurns,
  busy,
  currentGoal,
  onAction,
  onFollowUp,
  onNewSession,
  collapsed,
  onToggle,
}: {
  events: TaskEvent[];
  sessionTurns: SessionTurn[];
  busy: boolean;
  currentGoal: string | null;
  onAction: (fn: () => Promise<RuntimeSnapshot>) => void;
  onFollowUp: (goal: string) => void;
  onNewSession: () => void;
  collapsed: boolean;
  onToggle: () => void;
}) {
  const [followUp, setFollowUp] = useState("");
  const followUpRef = useRef<HTMLTextAreaElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);
  const latest = events.at(-1);
  const streamed = [...events].reverse().find((event) => event.phase === "planning" && event.title === "模型输出中");
  const terminalPhase = latest?.phase === "completed" || latest?.phase === "stopped" || latest?.phase === "error";
  // After a turn is archived into sessionTurns, avoid duplicating it as the live bubble.
  const latestArchived = sessionTurns.at(-1);
  const liveMatchesArchive = terminalPhase
    && !!latestArchived
    && latestArchived.id.startsWith(`${latest?.taskId || ""}:`);
  const activityEvents = liveMatchesArchive
    ? []
    : events.filter((event) => event.title !== "模型输出中" && event.taskId !== "local-submit").slice(-6);
  const canStop = !!latest && ["observing", "planning", "awaiting_user", "executing"].includes(latest.phase);
  const canFollowUp = !busy && (terminalPhase || (sessionTurns.length > 0 && events.length === 0) || latest?.phase === "paused");
  const flow = [
    { phase: "observing", label: "观察" },
    { phase: "planning", label: "决定" },
    { phase: "executing", label: "操作" },
    { phase: "completed", label: "验证" },
  ];
  const reached = (phase: string) =>
    events.some((event) => event.phase === phase)
    || (phase === "planning" && !!streamed)
    || (phase === "completed" && (latest?.phase === "completed" || latestArchived?.phase === "completed"));

  useEffect(() => {
    if (!canFollowUp || collapsed) return;
    const timer = window.setTimeout(() => followUpRef.current?.focus(), 180);
    return () => window.clearTimeout(timer);
  }, [canFollowUp, collapsed, sessionTurns.length, latest?.phase]);

  useEffect(() => {
    const node = listRef.current;
    if (!node) return;
    node.scrollTop = node.scrollHeight;
  }, [sessionTurns.length, events.length, latest?.detail, canFollowUp]);

  function submitFollowUp() {
    const value = followUp.trim();
    if (!value || busy) return;
    setFollowUp("");
    onFollowUp(value);
  }

  const statusLabel = latest?.phase === "completed" || latestArchived?.phase === "completed"
    ? "可继续"
    : events.length || busy
      ? "进行中"
      : sessionTurns.length
        ? "可继续"
        : "待命";

  return (
    <div className="workflow-panel">
      <div className="workflow-panel-heading">
        <div className="conversation-heading">
          <span className="section-kicker">AGENT</span>
          <h2>任务对话</h2>
        </div>
        <div className="conversation-tools">
          {!collapsed && (
            <span className={`phase-badge ${latest?.phase || latestArchived?.phase || ""}`}>
              {statusLabel}
            </span>
          )}
          <button
            className="collapse-button"
            onClick={onToggle}
            title={collapsed ? "展开对话" : "收起对话"}
          >
            {collapsed ? "→" : "←"}
          </button>
        </div>
      </div>

      <div className="workflow-flow">
        {flow.map((item) => (
          <span
            className={`${reached(item.phase) ? "reached" : ""} ${latest?.phase === item.phase ? "active" : ""}`}
            key={item.phase}
          >
            <i />
            {item.label}
          </span>
        ))}
      </div>

      <div className="workflow-panel-list" ref={listRef}>
        {sessionTurns.length === 0 && activityEvents.length === 0 && !busy ? (
          <div className="empty-state">
            <ScanSearch size={20} />
            <span>输入任务后，Agent 会在这里回复。</span>
          </div>
        ) : (
          <>
            {sessionTurns.map((turn) => (
              <div key={turn.id} className="turn-block">
                <div className="turn-user">
                  <span className="turn-role">你</span>
                  <p>{turn.goal}</p>
                </div>
                <div className={`workflow-panel-item message-${turn.phase}`}>
                  <div className="workflow-panel-line">
                    <span className="timeline-icon">
                      {turn.phase === "completed" ? <Check size={13} /> : <AlertTriangle size={13} />}
                    </span>
                  </div>
                  <div className="workflow-panel-content">
                    <div className="timeline-title">
                      <strong>{turn.phase === "completed" ? "输出完毕" : turn.phase === "stopped" ? "已中止" : "需要处理"}</strong>
                    </div>
                    <p>{turn.result}</p>
                  </div>
                </div>
              </div>
            ))}

            {busy && currentGoal && !liveMatchesArchive && (
              <div className="turn-user turn-user-live">
                <span className="turn-role">你</span>
                <p>{currentGoal}</p>
              </div>
            )}

            {activityEvents.map((event, index) => (
              <div className={`workflow-panel-item message-${event.phase} ${index === activityEvents.length - 1 ? "current" : ""}`} key={`${event.taskId}-${event.phase}-${event.title}`}>
                <div className="workflow-panel-line">
                  <span className="timeline-icon">
                    {event.phase === "completed" ? <Check size={13} /> : event.phase === "error" ? <AlertTriangle size={13} /> : <Activity size={13} />}
                  </span>
                </div>
                <div className="workflow-panel-content">
                  <div className="timeline-title">
                    <strong>{event.title}</strong>
                    <time>刚刚</time>
                  </div>
                  <p>{event.detail}</p>
                  {canStop && index === activityEvents.length - 1 && (
                    <button className="stop-inline pressable" onClick={() => onAction(bridge.stop)}>
                      <CircleStop size={13} /> 中止任务
                    </button>
                  )}
                </div>
              </div>
            ))}
          </>
        )}
      </div>

      {canFollowUp && !collapsed && (
        <div className="follow-up">
          <div className="follow-up-head">
            <MessageSquarePlus size={14} />
            <strong>继续对话</strong>
            <button type="button" className="follow-up-reset" onClick={onNewSession} title="清空会话并开始新任务">
              新会话
            </button>
          </div>
          <div className="follow-up-composer">
            <textarea
              ref={followUpRef}
              value={followUp}
              rows={2}
              placeholder="基于当前结果，输入下一步指令…"
              onChange={(event) => setFollowUp(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submitFollowUp();
                }
              }}
            />
            <button
              type="button"
              className="follow-up-send pressable"
              disabled={!followUp.trim() || busy}
              onClick={submitFollowUp}
              title="发送后续指令"
            >
              <ArrowUp size={15} />
            </button>
          </div>
          <small>Enter 发送 · Shift+Enter 换行</small>
        </div>
      )}
    </div>
  );
}

function BotStage({
  phase,
  busy,
  result,
  status,
  detail,
  stepCount,
}: {
  phase: string;
  busy: boolean;
  result?: string;
  status: string;
  detail: string;
  stepCount: number;
}) {
  const terminal = phase === "completed" || phase === "stopped" || phase === "error";
  const caption =
    phase === "completed" ? "任务完成"
      : phase === "stopped" ? "任务已中止"
        : phase === "error" ? "需要处理"
          : busy ? "baodou 正在工作"
            : "准备就绪";
  const sub =
    busy && !terminal
      ? "正在观察、规划并执行"
      : terminal
        ? phase === "completed" ? "结果已就绪" : "可重新发起任务"
        : "输入任务后，Agent 会在这里回应";

  return (
    <div className={`bot-stage ${busy && !terminal ? "is-active" : ""} ${terminal ? "is-terminal" : ""}`}>
      <div className="computer-use-status">
        <div className="computer-use-status-copy">
          <span className={`status-pip ${busy && !terminal ? "live" : ""}`} />
          <div>
            <small>{busy && !terminal ? `COMPUTER USE · STEP ${Math.max(stepCount, 1)}` : "COMPUTER USE"}</small>
            <strong>{status}</strong>
          </div>
        </div>
        <span className={`stage-phase ${phase}`}>{phase === "completed" ? "已验证" : phase === "error" ? "执行失败" : phase === "stopped" ? "已中止" : busy ? "运行中" : "待命"}</span>
      </div>
      <p className="computer-use-detail">{detail}</p>
      <div className="bot-orbit orbit-one" />
      <div className="bot-orbit orbit-two" />
      <div className="bubble-bot" aria-hidden>
        <span className="bot-eye eye-left" />
        <span className="bot-eye eye-right" />
        <span className="bot-mouth" />
      </div>
      <div className="bot-caption">
        <strong>{caption}</strong>
        <span>{sub}</span>
      </div>
      {result && (
        <div className="bot-result">
          <span className="result-mark"><Check size={12} /></span>
          <div>
            <small>执行结果</small>
            <p>{result}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function SettingsPage(props: {
  modelPathDraft: string; setModelPathDraft: (value: string) => void;
  modelFilePathDraft: string; setModelFilePathDraft: (value: string) => void;
  mmprojPathDraft: string; setMmprojPathDraft: (value: string) => void;
  llamaUrlDraft: string; setLlamaUrlDraft: (value: string) => void;
  saveModelPath: () => void; modelConfigMessage: string;
}) {
  return (
    <div className="settings-page">
      <div className="settings-heading">
        <span className="section-kicker">PREFERENCES</span>
        <h1>设置</h1>
        <p>配置本地模型运行时。修改后点击保存，下次启动自动使用。</p>
      </div>
      <div className="settings-form">
        <label>llama-server.exe 路径</label>
        <input value={props.modelPathDraft} onChange={(event) => props.setModelPathDraft(event.target.value)} />
        <label>模型 GGUF 路径</label>
        <input value={props.modelFilePathDraft} onChange={(event) => props.setModelFilePathDraft(event.target.value)} />
        <label>MMPROJ 路径</label>
        <input value={props.mmprojPathDraft} onChange={(event) => props.setMmprojPathDraft(event.target.value)} />
        <label>LLAMA URL</label>
        <input value={props.llamaUrlDraft} onChange={(event) => props.setLlamaUrlDraft(event.target.value)} />
        <button className="pressable" onClick={props.saveModelPath}>保存模型配置</button>
        {props.modelConfigMessage && <small>{props.modelConfigMessage}</small>}
      </div>
    </div>
  );
}

function SimplePage({ icon, title, detail }: { icon: ReactNode; title: string; detail: string }) {
  return (
    <div className="simple-page">
      <div className="simple-page-icon">{icon}</div>
      <span className="section-kicker">WORKSPACE</span>
      <h1>{title}</h1>
      <p>{detail}</p>
    </div>
  );
}

function Metric({ icon, name, value }: { icon: ReactNode; name: string; value: string }) {
  return (
    <div className="metric">
      <span>{icon}</span>
      <label>{name}</label>
      <strong title={value}>{value}</strong>
    </div>
  );
}

export default App;
