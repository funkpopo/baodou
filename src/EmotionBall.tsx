import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import type { PointerEvent } from "react";
import type { Phase } from "./types";

/**
 * Thin React host around the upstream emotion-ball engine.
 *
 * The actual SVG renderer, eye-ring data, emotion definitions and animation
 * state machine are loaded unchanged from public/vendor/emotion-ball.
 */

export interface EmotionDefinition {
  id: string;
  name: string;
  group: string;
  desc?: string;
  [key: string]: unknown;
}

export interface EmotionRegistrationResult {
  ok: boolean;
  id?: string;
  errors?: string[];
}

export interface EmotionImportResult {
  ok: boolean;
  added: number;
  errors: string[];
}

export interface EmotionGroup {
  key: string;
  name: string;
  en?: string;
}

export interface EmotionConfigApi {
  register: (config: EmotionDefinition) => EmotionRegistrationResult;
  get: (id: string) => EmotionDefinition | null;
  list: (group?: string) => EmotionDefinition[];
  groups: () => EmotionGroup[];
  exportConfig: () => string;
  importConfig: (
    config: string | EmotionDefinition | EmotionDefinition[],
  ) => EmotionImportResult;
}

export interface EmotionAIMessage {
  emotionId: string;
  tips?: string;
}

export interface EmotionStyle {
  sketch?: number;
  [key: string]: unknown;
}

export type EmotionEventName = "change" | "tips" | "error" | string;
export type EmotionEventListener = (payload: unknown) => void;

export interface EmotionBallHandle {
  readonly emotionId: string | null;
  readonly touring: boolean;
  on: (event: EmotionEventName, listener: EmotionEventListener) => EmotionBallHandle;
  off: (event: EmotionEventName, listener: EmotionEventListener) => EmotionBallHandle;
  setEmotion: (id: string, options?: { auto?: boolean }) => boolean;
  handleAIMessage: (message: EmotionAIMessage | string) => boolean;
  startTour: (ids: string[], interval?: number) => void;
  stopTour: () => void;
  resetIdle: () => void;
  setGaze: (x: number, y: number) => EmotionBallHandle;
  clearGaze: () => EmotionBallHandle;
  setStyle: (style: EmotionStyle) => EmotionBallHandle;
  spin: (turns?: number, direction?: -1 | 1) => EmotionBallHandle;
  burst: (count?: number) => EmotionBallHandle;
  bounce: () => EmotionBallHandle;
  registerEmotion: (config: EmotionDefinition) => EmotionRegistrationResult;
  setActive: (on: boolean) => void;
  replay: () => void;
  destroy: () => void;
}

export interface EmotionBallOptions {
  emotion?: string;
  fallbackId?: string;
  shape?: "blob" | "wedge" | "gem" | string;
  eyeScale?: number;
  autostart?: boolean;
  lite?: boolean;
  label?: string;
  idle?: boolean | Record<string, unknown>;
  color?: string;
  eyeColor?: string;
}

export interface EmotionBallRuntimeApi {
  config: EmotionConfigApi;
  create: (
    target: HTMLElement,
    options?: EmotionBallOptions,
  ) => EmotionBallHandle;
}

declare global {
  interface Window {
    EmotionBall?: EmotionBallRuntimeApi;
  }
}

export interface EmotionBallProps {
  phase: Phase | string;
  active?: boolean;
  size?: "stage" | "floating";
  label?: string;
}

function emotionIdForPhase(phase: Phase | string, active: boolean) {
  if (phase === "error") return "34";
  if (phase === "stopped") return "41";
  if (phase === "recognizing" || active) return "40";
  return "02";
}

export const EmotionBall = forwardRef<EmotionBallHandle, EmotionBallProps>(function EmotionBall(
  {
    phase,
    active = phase === "recognizing",
    size = "stage",
    label = "Baodou Emotion Ball",
  },
  ref,
) {
  const hostRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<EmotionBallHandle | null>(null);
  const boundsRef = useRef<DOMRectReadOnly | null>(null);
  const emotionId = emotionIdForPhase(phase, active);

  // Keep the entire upstream surface available to callers without exposing
  // the DOM host or making React own any of the engine's animation state.
  const publicHandle = useMemo<EmotionBallHandle>(() => {
    const handle: EmotionBallHandle = {
      get emotionId() {
        return engineRef.current?.emotionId ?? null;
      },
      get touring() {
        return engineRef.current?.touring ?? false;
      },
      on(event, listener) {
        engineRef.current?.on(event, listener);
        return handle;
      },
      off(event, listener) {
        engineRef.current?.off(event, listener);
        return handle;
      },
      setEmotion(id, options) {
        return engineRef.current?.setEmotion(id, options) ?? false;
      },
      handleAIMessage(message) {
        return engineRef.current?.handleAIMessage(message) ?? false;
      },
      startTour(ids, interval) {
        engineRef.current?.startTour(ids, interval);
      },
      stopTour() {
        engineRef.current?.stopTour();
      },
      resetIdle() {
        engineRef.current?.resetIdle();
      },
      setGaze(x, y) {
        engineRef.current?.setGaze(x, y);
        return handle;
      },
      clearGaze() {
        engineRef.current?.clearGaze();
        return handle;
      },
      setStyle(style) {
        engineRef.current?.setStyle(style);
        return handle;
      },
      spin(turns, direction) {
        engineRef.current?.spin(turns, direction);
        return handle;
      },
      burst(count) {
        engineRef.current?.burst(count);
        return handle;
      },
      bounce() {
        engineRef.current?.bounce();
        return handle;
      },
      registerEmotion(config) {
        return engineRef.current?.registerEmotion(config) ?? {
          ok: false,
          id: config.id,
          errors: ["EmotionBall 尚未挂载"],
        };
      },
      setActive(on) {
        engineRef.current?.setActive(on);
      },
      replay() {
        engineRef.current?.replay();
      },
      destroy() {
        const engine = engineRef.current;
        if (engine) {
          engine.destroy();
          engineRef.current = null;
        }
      },
    };
    return handle;
  }, []);

  useImperativeHandle(ref, () => publicHandle, [publicHandle]);

  // Keep one upstream instance for the lifetime of this surface so a
  // status-label change cannot reset gaze, springs or the current animation.
  useEffect(() => {
    const host = hostRef.current;
    const api = window.EmotionBall;
    if (!host || !api) {
      console.error("[Baodou] emotion-ball runtime was not loaded");
      return;
    }

    const engine = api.create(host, {
      emotion: emotionId,
      fallbackId: "02",
      shape: "blob",
      eyeScale: size === "floating" ? 1.5 : 1,
      autostart: true,
      lite: size === "floating",
      label,
    });
    engineRef.current = engine;

    const updateBounds = () => {
      boundsRef.current = host.getBoundingClientRect();
    };
    updateBounds();

    const resizeObserver = typeof ResizeObserver !== "undefined"
      ? new ResizeObserver(updateBounds)
      : null;
    resizeObserver?.observe(host);
    window.addEventListener("resize", updateBounds);

    // The upstream ticker is shared by all balls. Pause this instance while
    // its surface is hidden so the always-mounted floating window does not
    // keep consuming animation frames when it is not being displayed.
    let isIntersecting = true;
    const syncActivity = () => {
      engine.setActive(isIntersecting && document.visibilityState === "visible");
    };
    const visibilityObserver = typeof IntersectionObserver !== "undefined"
      ? new IntersectionObserver(([entry]) => {
          isIntersecting = entry.isIntersecting;
          syncActivity();
        })
      : null;
    visibilityObserver?.observe(host);
    document.addEventListener("visibilitychange", syncActivity);
    syncActivity();

    return () => {
      document.removeEventListener("visibilitychange", syncActivity);
      visibilityObserver?.disconnect();
      resizeObserver?.disconnect();
      window.removeEventListener("resize", updateBounds);
      boundsRef.current = null;
      engine.destroy();
      engineRef.current = null;
    };
  }, [size]);

  useEffect(() => {
    engineRef.current?.setEmotion(emotionId);
  }, [emotionId]);

  function updateGaze(event: PointerEvent<HTMLDivElement>) {
    const bounds = boundsRef.current ?? event.currentTarget.getBoundingClientRect();
    boundsRef.current ??= bounds;
    if (!bounds.width || !bounds.height) return;
    const x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
    const y = ((event.clientY - bounds.top) / bounds.height) * 2 - 1;
    engineRef.current?.setGaze(
      Math.max(-1, Math.min(1, x)),
      Math.max(-1, Math.min(1, y)),
    );
  }

  return (
    <div
      ref={hostRef}
      className={`emotion-ball-host emotion-ball-host--${size}`}
      data-animation-active={active ? "true" : "false"}
      data-emotion-id={emotionId}
      role="img"
      aria-label={label}
      onPointerMove={updateGaze}
      onPointerLeave={() => engineRef.current?.setGaze(0, 0)}
    />
  );
});

EmotionBall.displayName = "EmotionBall";
