import { useEffect, useMemo, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { CircleStop, Minimize2, Play, Sparkles, X } from "lucide-react";
import { bridge } from "./bridge";
import type { FloatingMessage, Phase, RuntimeSnapshot } from "./types";

const DEFAULT_GOAL = "帮我观察当前电脑界面，留意最要紧、最清楚的可见内容";

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
  message: "我在桌边呢，随时可以帮你看屏幕。",
  rounds: 0,
  skippedRounds: 0,
  requests: 0,
  metrics: null,
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
      message: "我先看一眼现在的屏幕…",
    }));
    try {
      await bridge.start(DEFAULT_GOAL);
      // The Rust host shows the companion only after it has captured a clean
      // desktop backdrop. Showing it here races that capture and can feed the
      // companion's own speech bubble back into the vision model.
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
          <span>Baodou</span>
          <small>DESKTOP COMPANION</small>
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
              ? "Baodou 正在陪你观察，结果会实时刷新。"
              : "启动后 Baodou 会以悬浮精灵陪伴在桌面一侧。"}
          </p>
        </div>
      </section>

      {error && <div className="error-toast">{error}</div>}
    </main>
  );
}

const WAITING_TEXT = "我先看一眼现在的屏幕…";

function isWaitingSpeech(text: string) {
  const value = text.trim();
  return (
    !value ||
    value === WAITING_TEXT ||
    value.startsWith("我在桌边呢") ||
    value === "本地屏幕识别运行时已就绪"
  );
}

function floatingStatusLabel(phase: Phase, active: boolean) {
  if (phase === "error") return "需要处理";
  if (phase === "stopped") return "已暂停";
  if (active || phase === "recognizing") return "观察中";
  return "等待中";
}

function measureFloatingShell(shell: HTMLElement) {
  const styles = getComputedStyle(shell);
  const padX = Number.parseFloat(styles.paddingLeft) + Number.parseFloat(styles.paddingRight);
  const padY = Number.parseFloat(styles.paddingTop) + Number.parseFloat(styles.paddingBottom);
  const gap = Number.parseFloat(styles.columnGap || styles.gap) || 0;
  const pet = shell.querySelector<HTMLElement>(".floating-pet");
  const speech = shell.querySelector<HTMLElement>(".floating-speech");
  const petW = pet?.offsetWidth ?? 88;
  const petH = pet?.offsetHeight ?? 88;
  if (!speech) {
    return { width: Math.ceil(padX + petW), height: Math.ceil(padY + petH) };
  }

  // WebView's initial containing block is the current HWND. Measure inside a
  // wide offscreen host so the bubble can report its intrinsic size.
  const host = document.createElement("div");
  host.setAttribute("aria-hidden", "true");
  host.style.cssText =
    "position:absolute;left:-10000px;top:0;width:10000px;height:auto;visibility:hidden;pointer-events:none;";
  const probe = speech.cloneNode(true) as HTMLElement;
  probe.style.cssText =
    "position:static;display:flex;flex-direction:column;width:max-content;max-width:348px;height:auto;max-height:none;margin:0;overflow:visible;";
  const probeBody = probe.querySelector("p");
  if (probeBody instanceof HTMLElement) {
    probeBody.style.cssText =
      "width:max-content;max-width:328px;height:auto;overflow:visible;white-space:pre-wrap;";
  }
  host.appendChild(probe);
  document.body.appendChild(host);
  const speechW = Math.ceil(probe.offsetWidth);
  const speechH = Math.ceil(probe.offsetHeight);
  host.remove();

  return {
    width: Math.ceil(padX + petW + gap + speechW + 8),
    height: Math.ceil(padY + Math.max(petH, speechH) + 6),
  };
}

function FloatingApp() {
  const [message, setMessage] = useState<FloatingMessage>({
    text: "",
    phase: "idle",
    updatedAt: "",
  });
  const [active, setActive] = useState(false);
  const messageBodyRef = useRef<HTMLParagraphElement>(null);
  const shellRef = useRef<HTMLElement>(null);
  const lastSizeRef = useRef({ width: 0, height: 0 });
  const scheduleSizeRef = useRef<(force?: boolean) => void>(() => undefined);
  const currentWindow = getCurrentWindow();

  useEffect(() => {
    const messageBody = messageBodyRef.current;
    if (messageBody) {
      messageBody.scrollTop = messageBody.scrollHeight;
    }
  }, [message.text]);

  useEffect(() => {
    document.documentElement.classList.add("floating-mode");
    document.body.classList.add("floating-mode");
    document.title = "Baodou · Desktop Companion";

    let cleanupFloating: (() => void) | undefined;
    let cleanupRecognition: (() => void) | undefined;
    void bridge
      .onFloating((payload) => {
        setActive(payload.phase === "recognizing");
        setMessage({
          ...payload,
          text: isWaitingSpeech(payload.text) ? "" : payload.text,
        });
      })
      .then((unlisten) => {
        cleanupFloating = unlisten;
      });
    void bridge
      .onRecognition((event) => {
        setActive(event.phase === "recognizing");
        setMessage({
          text: isWaitingSpeech(event.detail) ? "" : event.detail,
          phase: event.phase,
          updatedAt: event.timestamp,
        });
      })
      .then((unlisten) => {
        cleanupRecognition = unlisten;
      });
    void bridge.runtime().then((runtime) => {
      setActive(runtime.phase === "recognizing");
    });
    return () => {
      cleanupFloating?.();
      cleanupRecognition?.();
    };
  }, []);

  useEffect(() => {
    const shell = shellRef.current;
    if (!shell) return;

    let frame = 0;
    let shrinkTimer = 0;
    const syncSize = (force = false) => {
      const { width, height } = measureFloatingShell(shell);
      const last = lastSizeRef.current;
      if (
        !force &&
        Math.abs(width - last.width) < 1 &&
        Math.abs(height - last.height) < 1
      ) {
        return;
      }
      const expanding = width > last.width + 0.5 || height > last.height + 0.5;
      lastSizeRef.current = { width, height };
      window.clearTimeout(shrinkTimer);
      if (force || expanding || last.width === 0) {
        void bridge.resizeFloating(width, height);
        return;
      }
      shrinkTimer = window.setTimeout(() => {
        void bridge.resizeFloating(width, height);
      }, 90);
    };
    const schedule = (force = false) => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => syncSize(force));
    };
    scheduleSizeRef.current = schedule;

    schedule(true);
    const observer = new ResizeObserver(() => schedule());
    observer.observe(shell);

    let visible = false;
    const visibilityTimer = window.setInterval(() => {
      void currentWindow.isVisible().then((next) => {
        if (next && !visible) {
          lastSizeRef.current = { width: 0, height: 0 };
          schedule(true);
        }
        visible = next;
      });
    }, 200);

    return () => {
      scheduleSizeRef.current = () => undefined;
      window.cancelAnimationFrame(frame);
      window.clearTimeout(shrinkTimer);
      window.clearInterval(visibilityTimer);
      observer.disconnect();
    };
  }, [currentWindow]);

  useEffect(() => {
    lastSizeRef.current = { width: 0, height: 0 };
    scheduleSizeRef.current(true);
  }, [message.text, message.phase]);

  function dragWindow(event: React.MouseEvent<HTMLElement>) {
    if (event.button !== 0 || (event.target as HTMLElement).closest("button")) return;
    void currentWindow.startDragging();
  }

  async function closeAndStop() {
    try {
      // Closing the companion is also the user's stop action. The runtime
      // command updates the shared state first and hides this window itself.
      await bridge.stop();
    } catch {
      void currentWindow.hide();
    }
  }

  return (
    <main
      ref={shellRef}
      className={`floating-shell ${active ? "is-active" : ""}`}
      data-tauri-drag-region
      onMouseDown={dragWindow}
    >
      {message.text ? (
        <div className="floating-speech" role="status" aria-live="polite">
          <div className="floating-speech-meta">
            <span className={`status-pip ${active ? "live" : ""}`} />
            <strong>{floatingStatusLabel(message.phase, active)}</strong>
            <button
              className="floating-close"
              onClick={() => void closeAndStop()}
              aria-label="关闭悬浮窗并停止识别"
            >
              <X size={11} />
            </button>
          </div>
          <p ref={messageBodyRef}>{message.text}</p>
        </div>
      ) : null}

      <div className="floating-pet">
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
      </div>
      <p className="computer-use-detail">{detail}</p>
      <div className="bot-orbit orbit-one" />
      <div className="bot-orbit orbit-two" />
      <div className="bubble-bot" aria-hidden>
        <span className="bot-eye eye-left" />
        <span className="bot-eye eye-right" />
        <span className="bot-mouth" />
      </div>
    </div>
  );
}

export default App;
