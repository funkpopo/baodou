import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { CircleStop, Minimize2, Play, Sparkles, X } from "lucide-react";
import { bridge } from "./bridge";
import type { FloatingMessage, RuntimeSnapshot } from "./types";

const DEFAULT_GOAL = "描述当前屏幕上的关键可见内容";

const initialRuntime: RuntimeSnapshot = {
  protocolVersion: "2.0.0",
  mode: "live screen recognition",
  phase: "idle",
  connected: true,
  inferenceBackend: "llama.cpp · local vision",
  device: "SYCL0 · Intel Arc",
  modelReady: false,
  taskId: null,
  goal: null,
  message: "本地屏幕识别运行时已就绪",
};

function resolveWindowMode(): "main" | "floating" {
  try {
    if (getCurrentWindow().label === "floating") {
      return "floating";
    }
  } catch {
    // Browser preview or non-Tauri host.
  }

  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    if (params.get("window") === "floating" || window.location.hash === "#floating") {
      return "floating";
    }
  }

  return "main";
}

function App() {
  if (resolveWindowMode() === "floating") {
    return <FloatingApp />;
  }

  return <MainApp />;
}

function MainApp() {
  const [runtime, setRuntime] = useState(initialRuntime);
  const [error, setError] = useState("");
  const active = runtime.phase === "recognizing";
  const currentWindow = getCurrentWindow();

  useEffect(() => {
    void bridge.runtime().then(setRuntime).catch(() => setError("无法连接本地运行时"));
    const timer = window.setInterval(() => {
      void bridge.runtime().then(setRuntime).catch(() => undefined);
    }, 800);
    let cleanup: (() => void) | undefined;
    void bridge
      .onRecognition((event) => {
        setRuntime((current) => ({
          ...current,
          message: event.detail,
          phase: event.phase,
          taskId: event.taskId || current.taskId,
        }));
      })
      .then((unlisten) => {
        cleanup = unlisten;
      });
    return () => {
      window.clearInterval(timer);
      cleanup?.();
    };
  }, []);

  const phaseLabel = useMemo(
    () =>
      ({
        idle: "待命",
        recognizing: "识别中",
        stopped: "已停止",
        error: "需要处理",
      }[runtime.phase] ?? runtime.phase),
    [runtime.phase],
  );

  async function start() {
    if (active) return;
    setError("");
    setRuntime((current) => ({
      ...current,
      phase: "recognizing",
      goal: DEFAULT_GOAL,
      message: "正在采集并识别屏幕内容…",
    }));
    try {
      await bridge.start(DEFAULT_GOAL);
      await bridge.showFloating();
      await currentWindow.hide();
    } catch (cause) {
      setError(String(cause));
      setRuntime((current) => ({ ...current, phase: "error" }));
    }
  }

  async function stop() {
    try {
      setRuntime(await bridge.stop());
      await bridge.hideFloating();
    } catch (cause) {
      setError(String(cause));
    }
  }

  function dragWindow(event: React.MouseEvent<HTMLElement>) {
    if (event.button !== 0 || (event.target as HTMLElement).closest("button")) return;
    void currentWindow.startDragging();
  }

  return (
    <main className="app-shell main-shell">
      <header className="topbar" data-tauri-drag-region onMouseDown={dragWindow}>
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={14} />
          </div>
          <span>baodou</span>
          <small>PORTABLE</small>
        </div>
        <div className="topbar-center">
          <span className={`traffic-dot ${runtime.modelReady ? "ready" : ""}`} />
          {runtime.modelReady ? "模型已连接" : "模型未就绪"}
        </div>
        <div className="window-tools">
          <button className="window-button" onClick={() => void currentWindow.minimize()} aria-label="最小化">
            <Minimize2 size={14} />
          </button>
          <button className="window-button close-button" onClick={() => void currentWindow.hide()} aria-label="隐藏到任务栏">
            <X size={14} />
          </button>
        </div>
      </header>

      <section className="launch-pane">
        <BotStage active={active} phase={runtime.phase} status={phaseLabel} detail={runtime.message} />

        <div className="launch-actions">
          {active ? (
            <button className="launch-button stop pressable" onClick={() => void stop()}>
              <CircleStop size={18} />
              停止
            </button>
          ) : (
            <button className="launch-button start pressable" onClick={() => void start()}>
              <Play size={18} />
              启动
            </button>
          )}
          <p className="launch-hint">
            {active
              ? "悬浮窗已显示，识别结果会实时刷新。"
              : "启动后显示悬浮精灵，停止后自动隐藏。"}
          </p>
        </div>
      </section>

      {error && <div className="error-toast">{error}</div>}
    </main>
  );
}

function FloatingApp() {
  const [message, setMessage] = useState<FloatingMessage>({
    text: "正在识别屏幕内容…",
    phase: "recognizing",
    updatedAt: "",
  });
  const [active, setActive] = useState(true);
  const speechRef = useRef<HTMLDivElement>(null);
  const messageBodyRef = useRef<HTMLParagraphElement>(null);
  const currentWindow = getCurrentWindow();

  useLayoutEffect(() => {
    // Let the new message paint first, then keep the newest recognition text
    // visible and fit the native transparent window around the speech bubble.
    const frame = window.requestAnimationFrame(() => {
      const messageBody = messageBodyRef.current;
      if (messageBody) {
        messageBody.scrollTop = messageBody.scrollHeight;
      }

      const speech = speechRef.current;
      if (speech) {
        // The pet occupies the right edge; account for it when fitting the
        // native transparent window to the naturally sized speech bubble.
        const width = Math.ceil(speech.getBoundingClientRect().width + 98);
        const height = Math.ceil(speech.getBoundingClientRect().height + 24);
        void bridge.resizeFloating(width, height).catch(() => undefined);
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, [message.text]);

  useEffect(() => {
    document.documentElement.classList.add("floating-mode");
    document.body.classList.add("floating-mode");
    document.title = "baodou floating";

    let cleanupFloating: (() => void) | undefined;
    let cleanupRecognition: (() => void) | undefined;
    void bridge
      .onFloating((payload) => {
        setMessage(payload);
        setActive(payload.phase === "recognizing");
      })
      .then((unlisten) => {
        cleanupFloating = unlisten;
      });
    void bridge
      .onRecognition((event) => {
        setMessage({
          text: event.detail,
          phase: event.phase,
          updatedAt: event.timestamp,
        });
        setActive(event.phase === "recognizing");
      })
      .then((unlisten) => {
        cleanupRecognition = unlisten;
      });
    void bridge.runtime().then((runtime) => {
      setActive(runtime.phase === "recognizing");
      if (runtime.message) {
        setMessage((current) => ({
          ...current,
          text: runtime.message,
          phase: runtime.phase,
        }));
      }
    });
    return () => {
      cleanupFloating?.();
      cleanupRecognition?.();
    };
  }, []);

  function dragWindow(event: React.MouseEvent<HTMLElement>) {
    if (event.button !== 0 || (event.target as HTMLElement).closest("button")) return;
    void currentWindow.startDragging();
  }

  async function hide() {
    try {
      await bridge.hideFloating();
    } catch {
      void currentWindow.hide();
    }
  }

  return (
    <main className={`floating-shell ${active ? "is-active" : ""}`} data-tauri-drag-region onMouseDown={dragWindow}>
      {/* Speech bubble sits outside the spirit body on a pure transparent canvas. */}
      <div ref={speechRef} className="floating-speech" role="status" aria-live="polite">
        <div className="floating-speech-meta">
          <span className={`status-pip ${active ? "live" : ""}`} />
          <strong>{active ? "识别中" : "已暂停"}</strong>
          <button className="floating-close" onClick={() => void hide()} aria-label="隐藏悬浮窗">
            <X size={11} />
          </button>
        </div>
        <p ref={messageBodyRef}>{message.text || "等待识别结果…"}</p>
      </div>

      <div className="floating-pet" aria-hidden>
        <div className="bot-orbit orbit-one" />
        <div className="bot-orbit orbit-two" />
        <div className="bubble-bot">
          <span className="bot-eye eye-left" />
          <span className="bot-eye eye-right" />
          <span className="bot-mouth" />
        </div>
      </div>
    </main>
  );
}

function BotStage({
  active,
  phase,
  status,
  detail,
}: {
  active: boolean;
  phase: string;
  status: string;
  detail: string;
}) {
  const terminal = phase === "stopped" || phase === "error";
  const caption = active ? "baodou 正在观察" : terminal ? (phase === "stopped" ? "识别已停止" : "需要处理") : "准备就绪";
  const sub = active
    ? "结果会同步到系统级悬浮窗"
    : terminal
      ? "可以再次启动继续识别"
      : "点击启动开始实时屏幕识别";

  return (
    <div className={`bot-stage ${active ? "is-active" : ""} ${terminal ? "is-terminal" : ""}`}>
      <div className="computer-use-status">
        <div className="computer-use-status-copy">
          <span className={`status-pip ${active ? "live" : ""}`} />
          <div>
            <small>LIVE SCREEN RECOGNITION</small>
            <strong>{status}</strong>
          </div>
        </div>
        <span className={`stage-phase ${phase}`}>
          {active ? "运行中" : phase === "stopped" ? "已停止" : phase === "error" ? "异常" : "待命"}
        </span>
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
    </div>
  );
}

export default App;
