import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { RuntimeSnapshot, TaskEvent } from "./types";

export const bridge = {
  runtime: () => invoke<RuntimeSnapshot>("get_runtime"),
  start: (goal: string, live: boolean, autoConfirm = false) =>
    invoke<string>("run_task", { request: { goal, live, autoConfirm } }),
  confirm: () => invoke<RuntimeSnapshot>("confirm_task"),
  pause: () => invoke<RuntimeSnapshot>("pause_runtime"),
  stop: () => invoke<RuntimeSnapshot>("stop_runtime"),
  onTask: (listener: (event: TaskEvent) => void): Promise<UnlistenFn> =>
    listen<TaskEvent>("task-event", (event) => listener(event.payload)),
};
