import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

function markFloatingMode() {
  try {
    const params = new URLSearchParams(window.location.search);
    const isFloating =
      params.get("window") === "floating" ||
      window.location.hash === "#floating" ||
      document.title.toLowerCase().includes("floating");
    if (isFloating) {
      document.documentElement.classList.add("floating-mode");
      document.body.classList.add("floating-mode");
    }
  } catch {
    // ignore
  }
}

markFloatingMode();

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode><App /></React.StrictMode>,
);
