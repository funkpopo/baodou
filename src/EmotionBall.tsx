import { useEffect, useRef } from "react";
import type { PointerEvent } from "react";
import type { Phase } from "./types";

/**
 * Thin React host around the upstream emotion-ball engine.
 *
 * The actual SVG renderer, eye-ring data, emotion definitions and animation
 * state machine are loaded unchanged from public/vendor/emotion-ball.
 */

interface UpstreamEmotionEngine {
  setEmotion: (id: string) => boolean;
  setGaze: (x: number, y: number) => UpstreamEmotionEngine;
  destroy: () => void;
}

interface UpstreamEmotionBallApi {
  create: (
    target: HTMLElement,
    options: {
      emotion: string;
      fallbackId: string;
      shape: "blob";
      eyeScale: number;
      autostart: boolean;
      lite: boolean;
      label: string;
    },
  ) => UpstreamEmotionEngine;
}

declare global {
  interface Window {
    EmotionBall?: UpstreamEmotionBallApi;
  }
}

interface EmotionBallProps {
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

export function EmotionBall({
  phase,
  active = phase === "recognizing",
  size = "stage",
  label = "Baodou Emotion Ball",
}: EmotionBallProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<UpstreamEmotionEngine | null>(null);
  const emotionId = emotionIdForPhase(phase, active);

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

    return () => {
      engine.destroy();
      engineRef.current = null;
    };
  }, [label, size]);

  useEffect(() => {
    engineRef.current?.setEmotion(emotionId);
  }, [emotionId]);

  function updateGaze(event: PointerEvent<HTMLDivElement>) {
    const bounds = event.currentTarget.getBoundingClientRect();
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
      data-emotion-id={emotionId}
      role="img"
      aria-label={label}
      onPointerMove={updateGaze}
      onPointerLeave={() => engineRef.current?.setGaze(0, 0)}
    />
  );
}

