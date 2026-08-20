import { useEffect, useMemo, useRef, useState } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  ArrowLeft,
  Check,
  CircleStop,
  Minimize2,
  Play,
  Save,
  Settings,
  Sparkles,
  X,
} from "lucide-react";
import { bridge } from "./bridge";
import { EmotionBall } from "./EmotionBall";
import type { FloatingMessage, ModelConfig, Phase, RuntimeSnapshot } from "./types";

const DEFAULT_GOAL = "帮我观察当前电脑界面，留意最要紧、最清楚的可见内容";

const initialRuntime: RuntimeSnapshot = {
  protocolVersion: "2.0.0",
  mode: "live screen recognition",
  phase: "idle",
  connected: true,
  inferenceBackend: "configured local vision service",
  device: "configured device",
  modelReady: false,
  modelStatus: "unconfigured",
  modelProgress: null,
  modelDetail: "请在设置中填写推理服务、模型和 mmproj 路径。",
  taskId: null,
  goal: null,
  message: "我在桌边呢，随时可以帮你看屏幕。",
  rounds: 0,
  skippedRounds: 0,
  requests: 0,
  metrics: null,
};

type MainView = "home" | "settings";

interface ModelConfigDraft {
  serverPath: string;
  modelPath: string;
  mmprojPath: string;
  llamaUrl: string;
  nGpuLayers: string;
  batchSize: string;
  ubatchSize: string;
  flashAttn: boolean;
  warmup: boolean;
  multiImageInput: boolean;
}

const emptyModelConfigDraft: ModelConfigDraft = {
  serverPath: "",
  modelPath: "",
  mmprojPath: "",
  llamaUrl: "",
  nGpuLayers: "",
  batchSize: "",
  ubatchSize: "",
  flashAttn: false,
  warmup: false,
  multiImageInput: false,
};

function modelConfigToDraft(config: ModelConfig): ModelConfigDraft {
  return {
    serverPath: config.serverPath,
    modelPath: config.modelPath,
    mmprojPath: config.mmprojPath,
    llamaUrl: config.llamaUrl,
    nGpuLayers: config.nGpuLayers == null ? "" : String(config.nGpuLayers),
    batchSize: config.batchSize == null ? "" : String(config.batchSize),
    ubatchSize: config.ubatchSize == null ? "" : String(config.ubatchSize),
    flashAttn: config.flashAttn ?? false,
    warmup: config.warmup ?? false,
    multiImageInput: config.multiImageInput ?? false,
  };
}

function draftToModelConfig(draft: ModelConfigDraft): ModelConfig {
  const optionalInteger = (value: string) => {
    const trimmed = value.trim();
    return trimmed ? Number.parseInt(trimmed, 10) : null;
  };

  return {
    serverPath: draft.serverPath.trim(),
    modelPath: draft.modelPath.trim(),
    mmprojPath: draft.mmprojPath.trim(),
    llamaUrl: draft.llamaUrl.trim(),
    nGpuLayers: optionalInteger(draft.nGpuLayers),
    batchSize: optionalInteger(draft.batchSize),
    ubatchSize: optionalInteger(draft.ubatchSize),
    flashAttn: draft.flashAttn,
    warmup: draft.warmup,
    multiImageInput: draft.multiImageInput,
  };
}

function errorMessage(cause: unknown) {
  return cause instanceof Error ? cause.message : String(cause);
}

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
  const [view, setView] = useState<MainView>("home");
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
          <span className={`traffic-dot model-${runtime.modelStatus}`} />
          {modelStatusLabel(runtime.modelStatus)}
        </div>
        <div className="window-tools">
          <button
            className={`window-button settings-button ${view === "settings" ? "is-active" : ""}`}
            onClick={() => {
              setError("");
              setView((current) => (current === "home" ? "settings" : "home"));
            }}
            aria-label={view === "settings" ? "返回主界面" : "打开模型配置"}
            title={view === "settings" ? "返回主界面" : "模型配置"}
          >
            {view === "settings" ? <ArrowLeft size={14} /> : <Settings size={14} />}
          </button>
          <button className="window-button" onClick={() => void currentWindow.minimize()} aria-label="最小化">
            <Minimize2 size={14} />
          </button>
          <button className="window-button close-button" onClick={() => void currentWindow.hide()} aria-label="隐藏到任务栏">
            <X size={14} />
          </button>
        </div>
      </header>

      {view === "settings" ? (
        <ModelSettingsPage modelReady={runtime.modelReady} onBack={() => setView("home")} />
      ) : (
        <section className="launch-pane">
          <BotStage active={active} phase={runtime.phase} status={phaseLabel} detail={runtime.message} />
          <ModelLoadStatus runtime={runtime} />

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
      )}

      {error && <div className="error-toast">{error}</div>}
    </main>
  );
}

function modelStatusLabel(status: string) {
  return (
    {
      unconfigured: "未配置",
      starting: "启动中",
      loading: "加载中",
      warming: "预热中",
      ready: "已就绪",
      reconnecting: "重连中",
      error: "模型错误",
    }[status] ?? "状态未知"
  );
}

function ModelLoadStatus({ runtime }: { runtime: RuntimeSnapshot }) {
  const progress = runtime.modelProgress;
  const status = runtime.modelStatus;
  const hasProgress = typeof progress === "number" && Number.isFinite(progress);
  const progressLabel = hasProgress ? `${Math.round(progress)}%` : "等待服务日志";

  return (
    <section
      className={`model-load-status model-status-${status} ${hasProgress ? "" : "is-indeterminate"}`}
      aria-live="polite"
    >
      <div className="model-load-status-head">
        <div>
          <span className="model-load-status-kicker">MODEL SERVICE</span>
          <strong>{modelStatusLabel(status)}</strong>
        </div>
        <span className="model-load-status-percent">{progressLabel}</span>
      </div>
      <div className="model-load-status-track" aria-label={`模型加载进度：${progressLabel}`}>
        <span
          className="model-load-status-fill"
          style={{ width: hasProgress ? `${Math.max(0, Math.min(100, progress ?? 0))}%` : "0%" }}
        />
      </div>
      <p title={runtime.modelDetail}>{runtime.modelDetail}</p>
    </section>
  );
}

function ModelSettingsPage({ modelReady, onBack }: { modelReady: boolean; onBack: () => void }) {
  const [draft, setDraft] = useState<ModelConfigDraft>(emptyModelConfigDraft);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    void bridge
      .modelConfig()
      .then((config) => {
        if (!mounted) return;
        setDraft(modelConfigToDraft(config));
        setLoading(false);
      })
      .catch((cause) => {
        if (!mounted) return;
        setError(`读取模型配置失败：${errorMessage(cause)}`);
        setLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  function updateText(field: keyof ModelConfigDraft, value: string) {
    setSaved(false);
    setDraft((current) => ({ ...current, [field]: value }));
  }

  function updateToggle(field: "flashAttn" | "warmup" | "multiImageInput", value: boolean) {
    setSaved(false);
    setDraft((current) => ({ ...current, [field]: value }));
  }

  function validate() {
    const required: Array<[string, string]> = [
      ["服务程序路径", draft.serverPath],
      ["模型文件路径", draft.modelPath],
      ["mmproj 文件路径", draft.mmprojPath],
      ["接口地址", draft.llamaUrl],
    ];
    const missing = required.find(([, value]) => !value.trim());
    if (missing) return `${missing[0]}不能为空`;

    const integerFields: Array<[string, string]> = [
      ["GPU 层数", draft.nGpuLayers],
      ["批处理大小", draft.batchSize],
      ["微批处理大小", draft.ubatchSize],
    ];
    const invalid = integerFields.find(
      ([, value]) => value.trim() && !/^-?\d+$/.test(value.trim()),
    );
    return invalid ? `${invalid[0]}需要填写整数` : "";
  }

  async function save() {
    const validationError = validate();
    if (validationError) {
      setSaved(false);
      setError(validationError);
      return;
    }

    setSaving(true);
    setError("");
    try {
      const config = await bridge.saveModelConfig(draftToModelConfig(draft));
      setDraft(modelConfigToDraft(config));
      setSaved(true);
    } catch (cause) {
      setSaved(false);
      setError(`保存模型配置失败：${errorMessage(cause)}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="settings-page" aria-labelledby="settings-title">
      <div className="settings-heading">
        <button className="settings-back pressable" onClick={onBack} aria-label="返回主界面">
          <ArrowLeft size={15} />
        </button>
        <div className="settings-heading-copy">
          <span className="settings-kicker">MODEL / RUNTIME</span>
          <h1 id="settings-title">模型配置</h1>
          <p>配置本地视觉服务与模型文件，保存后会自动重新连接。</p>
        </div>
        <div className={`settings-live ${modelReady ? "ready" : ""}`}>
          <span className="traffic-dot" />
          {modelReady ? "服务在线" : "等待服务"}
        </div>
      </div>

      {loading ? (
        <div className="settings-loading">
          <span className="settings-loading-dot" />
          正在读取本地配置…
        </div>
      ) : (
        <>
          <form
            id="model-settings-form"
            className="settings-form"
            onSubmit={(event) => {
              event.preventDefault();
              void save();
            }}
          >
          <section className="settings-section">
            <div className="settings-section-heading">
              <span className="settings-section-index">01</span>
              <div>
                <h2>服务连接</h2>
                <p>llama-server 的启动程序与 OpenAI 兼容接口。</p>
              </div>
            </div>
            <label className="settings-field settings-field-wide">
              <span>接口地址</span>
              <input
                value={draft.llamaUrl}
                onChange={(event) => updateText("llamaUrl", event.target.value)}
                placeholder="http://127.0.0.1:8765/v1/chat/completions"
                spellCheck={false}
                autoComplete="off"
                disabled={saving}
              />
              <small>远程地址也可以使用，但本地模型自动启动只适用于本机服务。</small>
            </label>
            <label className="settings-field settings-field-wide">
              <span>llama-server 程序</span>
              <input
                value={draft.serverPath}
                onChange={(event) => updateText("serverPath", event.target.value)}
                placeholder="选择 llama-server 可执行文件"
                spellCheck={false}
                autoComplete="off"
                disabled={saving}
              />
            </label>
          </section>

          <section className="settings-section">
            <div className="settings-section-heading">
              <span className="settings-section-index">02</span>
              <div>
                <h2>视觉资源</h2>
                <p>填写你选择的主模型与匹配的多模态投影文件。</p>
              </div>
            </div>
            <label className="settings-field settings-field-wide">
              <span>模型文件</span>
              <input
                value={draft.modelPath}
                onChange={(event) => updateText("modelPath", event.target.value)}
                placeholder="选择主模型文件（GGUF）"
                spellCheck={false}
                autoComplete="off"
                disabled={saving}
              />
            </label>
            <label className="settings-field settings-field-wide">
              <span>mmproj 文件</span>
              <input
                value={draft.mmprojPath}
                onChange={(event) => updateText("mmprojPath", event.target.value)}
                placeholder="选择多模态投影文件（mmproj）"
                spellCheck={false}
                autoComplete="off"
                disabled={saving}
              />
            </label>
          </section>

          <section className="settings-section settings-section-last">
            <div className="settings-section-heading">
              <span className="settings-section-index">03</span>
              <div>
                <h2>性能选项</h2>
                <p>留空使用 llama-server 默认值；上下文长度由运行时固定管理。</p>
              </div>
            </div>
            <div className="settings-grid">
              <label className="settings-field">
                <span>GPU 层数</span>
                <input
                  inputMode="numeric"
                  value={draft.nGpuLayers}
                  onChange={(event) => updateText("nGpuLayers", event.target.value)}
                  placeholder="默认"
                  disabled={saving}
                />
              </label>
              <label className="settings-field">
                <span>批处理大小</span>
                <input
                  inputMode="numeric"
                  value={draft.batchSize}
                  onChange={(event) => updateText("batchSize", event.target.value)}
                  placeholder="默认"
                  disabled={saving}
                />
              </label>
              <label className="settings-field">
                <span>微批处理大小</span>
                <input
                  inputMode="numeric"
                  value={draft.ubatchSize}
                  onChange={(event) => updateText("ubatchSize", event.target.value)}
                  placeholder="默认"
                  disabled={saving}
                />
              </label>
            </div>
            <div className="settings-toggles">
              <label className="settings-toggle">
                <input
                  type="checkbox"
                  checked={draft.flashAttn}
                  onChange={(event) => updateToggle("flashAttn", event.target.checked)}
                  disabled={saving}
                />
                <span className="settings-switch" />
                <span>
                  <strong>Flash Attention</strong>
                  <small>减少注意力计算的显存开销</small>
                </span>
              </label>
              <label className="settings-toggle">
                <input
                  type="checkbox"
                  checked={draft.warmup}
                  onChange={(event) => updateToggle("warmup", event.target.checked)}
                  disabled={saving}
                />
                <span className="settings-switch" />
                <span>
                  <strong>启动预热</strong>
                  <small>启动时预热模型视觉路径，降低首次识别延迟但会延长启动时间</small>
                </span>
              </label>
              <label className="settings-toggle">
                <input
                  type="checkbox"
                  checked={draft.multiImageInput}
                  onChange={(event) => updateToggle("multiImageInput", event.target.checked)}
                  disabled={saving}
                />
                <span className="settings-switch" />
                <span>
                  <strong>多图输入</strong>
                  <small>同时发送缩略图与局部裁剪，默认关闭</small>
                </span>
              </label>
            </div>
          </section>

          </form>

          <div className="settings-actions">
            <div className="settings-feedback" role="status" aria-live="polite">
              {error ? <span className="settings-feedback-error">{error}</span> : null}
              {!error && saved ? (
                <span className="settings-feedback-success">
                  <Check size={13} /> 配置已保存，服务正在重新连接
                </span>
              ) : null}
            </div>
            <button
              className="settings-save pressable"
              type="submit"
              form="model-settings-form"
              disabled={saving}
            >
              {saving ? <span className="button-spinner" /> : <Save size={15} />}
              {saving ? "正在保存" : "保存并应用"}
            </button>
          </div>
        </>
      )}
    </section>
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
  const hasLiveEventRef = useRef(false);
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
    const syncRuntimePhase = () => {
      void bridge.runtime().then((runtime) => {
        if (hasLiveEventRef.current) return;
        setActive(runtime.phase === "recognizing");
        const runtimeText = isWaitingSpeech(runtime.message) ? "" : runtime.message;
        setMessage((current) => {
          if (current.phase === runtime.phase && current.text === runtimeText) return current;
          return { ...current, phase: runtime.phase, text: runtimeText };
        });
      });
    };
    void bridge
      .onFloating((payload) => {
        hasLiveEventRef.current = true;
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
        hasLiveEventRef.current = true;
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
    // The floating window is created once and can remain hidden between
    // sessions. Poll until the first live event so a slow webview listener
    // cannot miss the initial recognizing phase.
    syncRuntimePhase();
    const runtimeTimer = window.setInterval(syncRuntimePhase, 800);
    return () => {
      cleanupFloating?.();
      cleanupRecognition?.();
      window.clearInterval(runtimeTimer);
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
        <EmotionBall phase={message.phase} active={active} size="floating" label="Baodou 悬浮精灵" />
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
      <EmotionBall phase={phase} active={active} size="stage" label={`Baodou ${status}`} />
    </div>
  );
}

export default App;
