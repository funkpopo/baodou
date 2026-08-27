// Shared types between the React frontends and the Tauri Rust runtime.
// All structs on the Rust side serialize with `#[serde(rename_all = "camelCase")]`.

/** Lifecycle phase. `paused` = recognizing but temporarily sleeping (1.2/2.4). */
export type Phase =
  | "idle"
  | "recognizing"
  | "paused"
  | "stopped"
  | "error";

/**
 * 1.1: lightweight semantic hint emitted by the Rust side.
 * Maps to EmotionBall emotions/actions in `HINT_EMOTION` / `HINT_ACTION`.
 */
export type EmotionHint =
  | "neutral"
  | "alert"
  | "message"
  | "progress"
  | "unclear"
  | "focused"
  | "error"
  | "rest";

export interface RecognitionEvent {
  taskId: string;
  phase: Phase;
  title: string;
  detail: string;
  timestamp: string;
  requiresConfirmation: boolean;
  complete: boolean;
  ok: boolean;
  /** 1.1: optional emotion hint driving the pet's expression. */
  emotionHint?: EmotionHint | string | null;
}

export interface FloatingMessage {
  text: string;
  phase: Phase;
  updatedAt: string;
  /** 1.1: optional emotion hint driving the pet's expression. */
  emotionHint?: EmotionHint | string | null;
}

export interface OpsMetrics {
  captureMs: number | null;
  encodeMs: number | null;
  firstTokenMs: number | null;
  firstContentTokenMs: number | null;
  finishReason: string | null;
  generateMs: number | null;
  totalMs: number | null;
  promptTokens: number | null;
  completionTokens: number | null;
  readability: string | null;
  inputKind: string | null;
  server: string | null;
  error: string | null;
}

export interface RuntimeSnapshot {
  protocolVersion: string;
  mode: string;
  phase: Phase;
  connected: boolean;
  inferenceBackend: string;
  device: string;
  modelReady: boolean;
  modelStatus: string;
  modelProgress: number | null;
  modelDetail: string;
  taskId: string | null;
  goal: string | null;
  message: string;
  /** 2.4: true while the recognition loop is sleeping for N minutes. */
  paused?: boolean;
  rounds: number;
  skippedRounds: number;
  requests: number;
  metrics: OpsMetrics | null;
}

export interface ModelConfig {
  serverPath: string;
  modelPath: string;
  mmprojPath: string;
  llamaUrl: string;
  nGpuLayers?: number | null;
  batchSize?: number | null;
  ubatchSize?: number | null;
  flashAttn?: boolean;
  warmup?: boolean;
  multiImageInput?: boolean;
}

export interface SessionHistory {
  id: string;
  goal: string;
  latestResult: string;
  startedAt: string;
  updatedAt: string;
}
