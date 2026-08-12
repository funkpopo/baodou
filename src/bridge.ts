import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { FloatingMessage, ModelConfig, RecognitionEvent, RuntimeSnapshot, SessionHistory } from "./types";

export const bridge = {
  runtime: () => invoke<RuntimeSnapshot>("get_runtime"),
  modelConfig: () => invoke<ModelConfig>("get_model_config"),
  saveModelConfig: (config: ModelConfig) => invoke<ModelConfig>("set_model_config", { config }),
  start: (goal?: string) => invoke<string>("run_task", { request: { goal: goal ?? null } }),
  stop: () => invoke<RuntimeSnapshot>("stop_runtime"),
  sessionHistory: () => invoke<SessionHistory[]>("get_session_history"),
  portablePaths: () =>
    invoke<{ root: string; dataDir: string; configPath: string; databasePath: string }>("get_portable_paths"),
  showFloating: () => invoke<void>("show_floating_window"),
  hideFloating: () => invoke<void>("hide_floating_window"),
  onRecognition: (listener: (event: RecognitionEvent) => void): Promise<UnlistenFn> =>
    listen<RecognitionEvent>("recognition-event", (event) => listener(event.payload)),
  onFloating: (listener: (message: FloatingMessage) => void): Promise<UnlistenFn> =>
    listen<FloatingMessage>("floating-message", (event) => listener(event.payload)),
};
