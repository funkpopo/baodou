import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type {
  FloatingMessage,
  ModelConfig,
  RecognitionEvent,
  RuntimeSnapshot,
  SessionHistory,
} from "./types";

export const bridge = {
  // ── Runtime / recognition ────────────────────────────────────────────────
  runtime: () => invoke<RuntimeSnapshot>("get_runtime"),
  /** 2.1: push-based snapshot updates (replaces polling). */
  onRuntime: (listener: (snapshot: RuntimeSnapshot) => void): Promise<UnlistenFn> =>
    listen<RuntimeSnapshot>("runtime-changed", (event) => listener(event.payload)),

  modelConfig: () => invoke<ModelConfig>("get_model_config"),
  saveModelConfig: (config: ModelConfig) =>
    invoke<ModelConfig>("set_model_config", { config }),
  start: (goal?: string) =>
    invoke<string>("run_task", { request: { goal: goal ?? null } }),
  stop: () => invoke<RuntimeSnapshot>("stop_runtime"),
  /** 1.2 / 2.4: pause the recognition loop for N minutes (window stays visible). */
  pauseRecognition: (minutes?: number) =>
    invoke<RuntimeSnapshot>("pause_recognition", { minutes: minutes ?? null }),
  /** 1.2 / 2.4: resume ahead of schedule. */
  resumeRecognition: () => invoke<RuntimeSnapshot>("resume_recognition"),
  /** 1.2: floating double-click — peek now, or restart with the last goal. */
  companionPeek: () => invoke<string>("companion_peek"),

  sessionHistory: () => invoke<SessionHistory[]>("get_session_history"),
  portablePaths: () =>
    invoke<{
      root: string;
      dataDir: string;
      configPath: string;
      databasePath: string;
    }>("get_portable_paths"),

  // ── Windows ──────────────────────────────────────────────────────────────
  showFloating: () => invoke<void>("show_floating_window"),
  hideFloating: () => invoke<void>("hide_floating_window"),
  resizeFloating: (width: number, height: number) =>
    invoke<void>("resize_floating_window", { width, height }),
  /** 1.2: bring the main window back (floating single-click). */
  focusMain: () => invoke<void>("focus_main_window"),
  /** 1.2: right-click menu → quit the whole app (stops the model). */
  exitApp: () => invoke<void>("exit_app"),

  // ── Events ───────────────────────────────────────────────────────────────
  onRecognition: (
    listener: (event: RecognitionEvent) => void,
  ): Promise<UnlistenFn> =>
    listen<RecognitionEvent>("recognition-event", (event) =>
      listener(event.payload),
    ),
  onFloating: (
    listener: (message: FloatingMessage) => void,
  ): Promise<UnlistenFn> =>
    listen<FloatingMessage>("floating-message", (event) =>
      listener(event.payload),
    ),
  /** 2.1: floating window visibility, pushed from Rust on show/hide. */
  onFloatingVisibility: (
    listener: (visible: boolean) => void,
  ): Promise<UnlistenFn> =>
    listen<boolean>("floating-visibility", (event) => listener(!!event.payload)),
};

export type Bridge = typeof bridge;
