export type Phase = "idle" | "observing" | "planning" | "awaiting_user" | "executing" | "paused" | "stopped" | "completed" | "error";

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
}

export interface ModelConfig {
  serverPath: string;
  modelPath: string;
  mmprojPath: string;
  llamaUrl: string;
}

export interface TaskEvent {
  taskId: string;
  phase: Phase;
  title: string;
  detail: string;
  timestamp: string;
  requiresConfirmation: boolean;
  complete: boolean;
  ok: boolean;
  raw?: unknown;
}
