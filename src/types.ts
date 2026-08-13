export type Phase = "idle" | "recognizing" | "stopped" | "error";

export interface OpsMetrics {
  captureMs: number | null;
  encodeMs: number | null;
  firstTokenMs: number | null;
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
  taskId: string | null;
  goal: string | null;
  message: string;
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
}

export interface RecognitionEvent {
  taskId: string;
  phase: Phase;
  title: string;
  detail: string;
  timestamp: string;
  requiresConfirmation: boolean;
  complete: boolean;
  ok: boolean;
}

export interface FloatingMessage { text: string; phase: Phase; updatedAt: string; }
export interface SessionHistory { id: string; goal: string; latestResult: string; startedAt: string; updatedAt: string; }
