export type GraphicsProfile = "webgl2" | "webgl" | "software";

/**
 * Detect the renderer capabilities exposed by the current WebView.
 *
 * This is deliberately a capability probe, not a request to use a discrete
 * GPU. WebView2 already performs hardware compositing by default and the OS
 * may choose an integrated GPU or a software fallback when a driver is
 * unavailable. `powerPreference: "default"` keeps that platform decision.
 */
export function detectGraphicsProfile(): GraphicsProfile {
  if (typeof document === "undefined") return "software";

  const canvas = document.createElement("canvas");
  const contextOptions: WebGLContextAttributes = {
    alpha: true,
    antialias: true,
    powerPreference: "default",
    preserveDrawingBuffer: false,
  };

  try {
    const webgl2 = canvas.getContext("webgl2", contextOptions);
    if (webgl2) return "webgl2";
  } catch {
    // A restricted WebView or a driver blacklist can reject context creation.
  }

  try {
    const webgl = canvas.getContext("webgl", contextOptions);
    if (webgl) return "webgl";
  } catch {
    // Keep the software profile. The SVG fallback remains fully functional.
  }

  return "software";
}

/** Apply renderer metadata before React mounts the UI. */
export function applyGraphicsHints() {
  if (typeof document === "undefined" || typeof window === "undefined") return;

  const root = document.documentElement;
  root.dataset.graphics = detectGraphicsProfile();
}
