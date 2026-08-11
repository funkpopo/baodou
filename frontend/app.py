"""Tkinter main window: task control, highlights, plan/confirm, metrics, diagnostics (Phase H).

Launch:
  python -m frontend.cli ui
  python -m frontend.app
"""

from __future__ import annotations

import contextlib
import json
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any

from core.cancel import get_global_token, install_signal_handlers
from core.config import AppConfig, load_config
from core.logging import get_logger, setup_logging
from PIL import ImageTk

from frontend.highlight import highlight_elements, highlight_preview_target, resize_for_preview
from frontend.session import UISession

_log = get_logger("frontend.app")


class BaodouApp:
    """Resident main window."""

    def __init__(self, config: AppConfig, *, mock: bool = True) -> None:
        self.config = config
        self.session = UISession(config, mock=mock, listener=self._on_session_event)
        self.root = tk.Tk()
        self.root.title(config.frontend.window_title)
        self.root.geometry("1180x780")
        self.root.minsize(960, 640)
        self._photo: ImageTk.PhotoImage | None = None
        self._poll_ms = max(200, int(config.frontend.refresh_ms))
        self._metrics_ms = max(500, int(config.frontend.metrics_interval_ms))
        self._building = True
        self._setup_style()
        self._build()
        self._building = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(self._poll_ms, self._poll_ui)
        self.root.after(self._metrics_ms, self._poll_metrics)
        # Initial idle status
        self._render_snapshot(self.session.snapshot().to_dict())

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Danger.TButton", foreground="#a00000")
        style.configure("Activity.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"))

    def _build(self) -> None:
        # --- Top: activity / privacy indicators ---
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="活动状态", style="Header.TLabel").pack(side=tk.LEFT)
        self.var_activity = tk.StringVar(value="idle")
        self.lbl_activity = ttk.Label(top, textvariable=self.var_activity, style="Activity.TLabel")
        self.lbl_activity.pack(side=tk.LEFT, padx=12)
        self.var_flags = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.var_flags).pack(side=tk.LEFT, padx=8)
        self.var_control = tk.StringVar(value="control: running")
        ttk.Label(top, textvariable=self.var_control).pack(side=tk.RIGHT)

        # --- Control bar ---
        bar = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="任务").pack(side=tk.LEFT)
        self.entry_goal = ttk.Entry(bar)
        self.entry_goal.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        self.entry_goal.insert(0, "点击搜索按钮")
        self.entry_goal.bind("<Return>", lambda _e: self._on_start())
        ttk.Button(bar, text="开始", command=self._on_start).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="仅预览", command=self._on_preview).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="暂停", command=self._on_pause).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="继续", command=self._on_resume).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="紧急停止", style="Danger.TButton", command=self._on_stop).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(bar, text="复位", command=self._on_reset).pack(side=tk.LEFT, padx=2)
        self.var_mock = tk.BooleanVar(value=self.session.mock)
        ttk.Checkbutton(
            bar, text="mock", variable=self.var_mock, command=self._on_mock_toggle
        ).pack(side=tk.LEFT, padx=8)
        self.var_auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="自动确认低风险", variable=self.var_auto).pack(side=tk.LEFT)

        # --- Main panes ---
        paned = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Left: preview + elements
        left = ttk.Frame(paned)
        paned.add(left, weight=3)
        prev_frame = ttk.LabelFrame(left, text="屏幕解读 / 目标高亮", padding=4)
        prev_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas = tk.Label(prev_frame, background="#1e1e1e", text="(无预览)")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        btn_row = ttk.Frame(left)
        btn_row.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(btn_row, text="刷新预览", command=self._on_refresh).pack(side=tk.LEFT, padx=2)
        el_frame = ttk.LabelFrame(left, text="识别到的 UI 元素", padding=4)
        el_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=False)
        self.tree = ttk.Treeview(
            el_frame,
            columns=("id", "type", "text", "click"),
            show="headings",
            height=8,
        )
        for col, w, title in (
            ("id", 140, "element_id"),
            ("type", 70, "type"),
            ("text", 160, "text"),
            ("click", 50, "可点"),
        ):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=w, anchor=tk.W)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        el_scroll = ttk.Scrollbar(el_frame, orient=tk.VERTICAL, command=self.tree.yview)
        el_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=el_scroll.set)
        corr_row = ttk.Frame(left)
        corr_row.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Button(corr_row, text="不是这个", command=self._on_reject).pack(side=tk.LEFT, padx=2)
        ttk.Button(corr_row, text="点这个", command=self._on_prefer).pack(side=tk.LEFT, padx=2)
        ttk.Button(corr_row, text="忽略该区域", command=self._on_ignore).pack(side=tk.LEFT, padx=2)
        ttk.Button(corr_row, text="清除修正", command=self._on_clear_corr).pack(
            side=tk.LEFT, padx=2
        )
        self.var_corr = tk.StringVar(value="修正: 0")
        ttk.Label(corr_row, textvariable=self.var_corr).pack(side=tk.RIGHT, padx=4)

        # Right: plan / confirm / metrics / diagnostics
        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        plan_frame = ttk.LabelFrame(right, text="当前计划 / 即将执行", padding=4)
        plan_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.txt_plan = scrolledtext.ScrolledText(plan_frame, height=10, wrap=tk.WORD)
        self.txt_plan.pack(fill=tk.BOTH, expand=True)

        conf_frame = ttk.LabelFrame(right, text="确认", padding=4)
        conf_frame.pack(side=tk.TOP, fill=tk.X, pady=4)
        self.var_pending = tk.StringVar(value="无待确认动作")
        ttk.Label(conf_frame, textvariable=self.var_pending, wraplength=360).pack(
            anchor=tk.W, pady=2
        )
        self.var_risk = tk.StringVar(value="风险: —")
        ttk.Label(conf_frame, textvariable=self.var_risk).pack(anchor=tk.W)
        cbtns = ttk.Frame(conf_frame)
        cbtns.pack(fill=tk.X, pady=4)
        ttk.Button(cbtns, text="确认执行", command=lambda: self._on_confirm(True)).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(cbtns, text="拒绝", command=lambda: self._on_confirm(False)).pack(
            side=tk.LEFT, padx=4
        )

        met_frame = ttk.LabelFrame(right, text="可观测性", padding=4)
        met_frame.pack(side=tk.TOP, fill=tk.X)
        self.var_metrics = tk.StringVar(value="—")
        ttk.Label(met_frame, textvariable=self.var_metrics, justify=tk.LEFT).pack(anchor=tk.W)

        err_frame = ttk.LabelFrame(right, text="最近错误", padding=4)
        err_frame.pack(side=tk.TOP, fill=tk.X)
        self.txt_errors = scrolledtext.ScrolledText(err_frame, height=4, wrap=tk.WORD)
        self.txt_errors.pack(fill=tk.BOTH, expand=True)

        if self.config.frontend.show_diagnostics:
            diag_frame = ttk.LabelFrame(right, text="开发者诊断", padding=4)
            diag_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=4)
            self.txt_diag = scrolledtext.ScrolledText(diag_frame, height=8, wrap=tk.WORD)
            self.txt_diag.pack(fill=tk.BOTH, expand=True)
            ttk.Button(diag_frame, text="刷新诊断", command=self._on_diag).pack(anchor=tk.E, pady=2)
        else:
            self.txt_diag = None

        # Status bar
        self.var_status = tk.StringVar(value="就绪 · dry_run · mock")
        ttk.Label(self.root, textvariable=self.var_status, padding=4).pack(
            side=tk.BOTTOM, fill=tk.X
        )

    # ------------------------------------------------------------------ actions
    def _on_mock_toggle(self) -> None:
        self.session.set_mock(bool(self.var_mock.get()))
        self._update_status_bar()

    def _on_start(self) -> None:
        goal = self.entry_goal.get().strip()
        if not goal:
            messagebox.showwarning("baodou", "请输入任务目标")
            return
        if self.session.is_busy():
            messagebox.showinfo("baodou", "任务进行中")
            return
        try:
            self.session.start_task(
                goal,
                execute=True,
                auto_confirm=bool(self.var_auto.get()),
                background=True,
            )
            self.var_status.set(f"运行中: {goal[:40]}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("baodou", str(exc))

    def _on_preview(self) -> None:
        goal = self.entry_goal.get().strip() or "描述当前屏幕"
        if self.session.is_busy():
            messagebox.showinfo("baodou", "任务进行中")
            return
        try:
            self.session.start_task(goal, execute=False, auto_confirm=False, background=True)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("baodou", str(exc))

    def _on_pause(self) -> None:
        self.session.pause("ui_pause")

    def _on_resume(self) -> None:
        self.session.resume("ui_resume")

    def _on_stop(self) -> None:
        if messagebox.askyesno("紧急停止", "确认紧急停止？将取消当前任务。"):
            self.session.emergency_stop("ui_emergency_stop")

    def _on_reset(self) -> None:
        self.session.reset_stop("ui_reset")
        self.var_status.set("已复位")

    def _on_confirm(self, accept: bool) -> None:
        if not self.session.confirm_pending(accept):
            messagebox.showinfo("baodou", "当前没有待确认动作")
            return
        self.var_pending.set("已提交确认" if accept else "已拒绝")

    def _selected_element_id(self) -> str | None:
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0], "values")
        return str(vals[0]) if vals else None

    def _on_reject(self) -> None:
        eid = self._selected_element_id()
        if not eid:
            messagebox.showinfo("baodou", "请先在列表中选择元素")
            return
        self.session.reject_element(eid, note="不是这个按钮")
        self.var_corr.set(f"修正: {len(self.session.corrections.items)}")

    def _on_prefer(self) -> None:
        eid = self._selected_element_id()
        if not eid:
            messagebox.showinfo("baodou", "请先在列表中选择元素")
            return
        self.session.prefer_element(eid, note="点击这里/这个")
        self.var_corr.set(f"修正: {len(self.session.corrections.items)}")

    def _on_ignore(self) -> None:
        eid = self._selected_element_id()
        if not eid:
            messagebox.showinfo("baodou", "请先选择要忽略区域的元素")
            return
        bbox = self.session.element_bbox(eid)
        if bbox is None:
            messagebox.showinfo("baodou", "找不到该元素 bbox")
            return
        self.session.ignore_region(bbox.x, bbox.y, bbox.width, bbox.height, note=f"忽略 {eid}")
        self.var_corr.set(f"修正: {len(self.session.corrections.items)}")

    def _on_clear_corr(self) -> None:
        self.session.clear_corrections()
        self.var_corr.set("修正: 0")

    def _on_refresh(self) -> None:
        def _work() -> None:
            try:
                self.session.refresh_observe(self.entry_goal.get().strip() or None)
            except Exception as exc:  # noqa: BLE001
                self.session.metrics.push_error(str(exc))
            finally:
                self.root.after(0, lambda: self._render_snapshot(self.session.snapshot().to_dict()))

        threading.Thread(target=_work, name="baodou-refresh", daemon=True).start()

    def _on_diag(self) -> None:
        if self.txt_diag is None:
            return
        bundle = self.session.diagnostics().to_dict()
        self.txt_diag.delete("1.0", tk.END)
        self.txt_diag.insert(tk.END, json.dumps(bundle, ensure_ascii=False, indent=2, default=str))

    def _on_close(self) -> None:
        if self.session.is_busy():
            if not messagebox.askyesno("退出", "任务进行中，确认退出？"):
                return
            self.session.emergency_stop("ui_close")
        self.root.destroy()

    # ------------------------------------------------------------------ render
    def _on_session_event(self, snap: dict[str, Any]) -> None:
        # Called from worker threads — marshal to UI thread
        with contextlib.suppress(Exception):
            self.root.after(0, lambda: self._render_snapshot(snap))

    def _poll_ui(self) -> None:
        with contextlib.suppress(Exception):
            self._render_snapshot(self.session.snapshot().to_dict())
        with contextlib.suppress(Exception):
            self.root.after(self._poll_ms, self._poll_ui)

    def _poll_metrics(self) -> None:
        with contextlib.suppress(Exception):
            m = self.session.metrics.snapshot()
            self.var_metrics.set(
                "采集 {cap} ms · 识别 {vis} ms · 模型 {mod} ms · 端到端 {e2e} ms\n"
                "队列 {q} (drop {d}) · CPU {cpu}% · RSS {rss} MB · GPU {gpu}".format(
                    cap=_fmt(m.capture_latency_ms),
                    vis=_fmt(m.vision_latency_ms),
                    mod=_fmt(m.model_latency_ms),
                    e2e=_fmt(m.end_to_end_ms),
                    q=m.queue_length,
                    d=m.queue_dropped,
                    cpu=_fmt(m.cpu_percent),
                    rss=_fmt(m.memory_rss_mb),
                    gpu=m.gpu_name or "—",
                )
            )
            if m.recent_errors:
                self.txt_errors.delete("1.0", tk.END)
                self.txt_errors.insert(tk.END, "\n".join(m.recent_errors[:8]))
        with contextlib.suppress(Exception):
            self.root.after(self._metrics_ms, self._poll_metrics)

    def _update_status_bar(self) -> None:
        snap = self.session.snapshot()
        bits = [
            snap.activity.phase.value,
            "dry_run" if snap.dry_run else "LIVE",
            "mock" if snap.mock else "live-backends",
            snap.control_state,
        ]
        if snap.busy:
            bits.insert(0, "busy")
        self.var_status.set(" · ".join(bits))

    def _render_snapshot(self, snap: dict[str, Any]) -> None:
        if self._building:
            return
        act = snap.get("activity") or {}
        phase = act.get("phase", "idle")
        self.var_activity.set(f"{phase} — {act.get('message', '')}")
        flags = []
        if act.get("capturing"):
            flags.append("📷 采集中")
        if act.get("recognizing"):
            flags.append("🔍 识别中")
        if act.get("inferring"):
            flags.append("🧠 推理中")
        if act.get("about_to_act"):
            flags.append("⚠️ 即将执行")
        if act.get("executing"):
            flags.append("🖱️ 执行中")
        self.var_flags.set("  ".join(flags) if flags else "（空闲）")
        self.var_control.set(f"control: {snap.get('control_state', 'running')}")
        self.var_corr.set(f"修正: {len(snap.get('corrections') or [])}")

        # Elements tree
        for item in self.tree.get_children():
            self.tree.delete(item)
        for el in snap.get("elements") or []:
            self.tree.insert(
                "",
                tk.END,
                values=(
                    el.get("element_id", ""),
                    el.get("type", ""),
                    el.get("text", ""),
                    "Y" if el.get("clickable") else "",
                ),
            )

        # Plan text
        lines: list[str] = []
        if snap.get("observation_text"):
            lines.append(f"【屏幕解读】\n{snap['observation_text'][:500]}\n")
        plan = snap.get("plan_summary")
        if plan:
            lines.append(
                f"【计划】goal={plan.get('goal', '')} steps={plan.get('step_count', plan.get('steps', '?'))} "
                f"risk_max={plan.get('risk_max', '')}\n"
            )
        for i, p in enumerate(snap.get("previews") or []):
            lines.append(
                f"  {i + 1}. [{p.get('risk', '')}] {p.get('summary', '')} "
                f"→ {p.get('target_element_id') or ''} "
                f"{'(需确认)' if p.get('requires_confirmation') else ''}"
            )
            if p.get("expected_impact"):
                lines.append(f"     影响: {p.get('expected_impact')}")
            for w in p.get("warnings") or []:
                lines.append(f"     ⚠ {w}")
        if snap.get("last_error"):
            lines.append(f"\n【错误】{snap['last_error']}")
        self.txt_plan.delete("1.0", tk.END)
        self.txt_plan.insert(tk.END, "\n".join(lines) if lines else "（尚无计划）")

        # Pending confirm
        pending = snap.get("pending_preview")
        if pending:
            self.var_pending.set(
                f"{pending.get('summary', '')}\n影响: {pending.get('expected_impact', '')}"
            )
            self.var_risk.set(
                f"风险: {pending.get('risk', '—')} · 目标: {pending.get('target_element_id') or pending.get('target_point') or '—'}"
            )
        else:
            self.var_pending.set("无待确认动作")
            self.var_risk.set("风险: —")

        self._update_preview_image(snap)
        self._update_status_bar()

    def _update_preview_image(self, snap: dict[str, Any]) -> None:
        image = self.session.get_last_image()
        vision = self.session.get_last_vision()
        frame = self.session.get_last_frame()
        if image is None:
            # Try mock blank
            return
        try:
            hi = set(snap.get("highlight_ids") or [])
            rejected = {
                c.get("element_id")
                for c in (snap.get("corrections") or [])
                if c.get("kind") == "reject_element" and c.get("element_id")
            }
            conv = frame.screen_to_image if frame is not None else None
            if vision is not None:
                result = self.session.get_last_result()
                pending = snap.get("pending_preview")
                if pending and result and result.previews:
                    # Find matching preview object
                    from core.models import ActionPreview

                    prev_obj = None
                    for p in result.previews:
                        if p.preview_id == pending.get("preview_id") or p.step_id == pending.get(
                            "step_id"
                        ):
                            prev_obj = p
                            break
                    if prev_obj is None and result.previews:
                        # synthesize from dict
                        try:
                            prev_obj = ActionPreview.model_validate(
                                {**pending, "summary": pending.get("summary") or "action"}
                            )
                        except Exception:  # noqa: BLE001
                            prev_obj = result.previews[0]
                    annotated = highlight_preview_target(
                        image, prev_obj, vision, screen_to_image=conv
                    )
                else:
                    annotated = highlight_elements(
                        image,
                        vision.elements,
                        screen_to_image=conv,
                        highlight_ids=hi,
                        rejected_ids=rejected,
                    )
            else:
                annotated = image.convert("RGB")
            annotated = resize_for_preview(
                annotated,
                self.config.frontend.preview_max_width,
                self.config.frontend.preview_max_height,
            )
            self._photo = ImageTk.PhotoImage(annotated)
            self.canvas.configure(image=self._photo, text="")
        except Exception as exc:  # noqa: BLE001
            self.canvas.configure(text=f"预览失败: {exc}")

    def run(self) -> None:
        self.root.mainloop()


def _fmt(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def run_app(
    config_path: str | None = None,
    *,
    mock: bool = True,
    log_level: str | None = None,
) -> int:
    cfg = load_config(config_path)
    level = log_level or cfg.app.log_level
    setup_logging(
        level=level,
        json_logs=cfg.app.log_json,
        log_dir=cfg.app.log_dir,
        project_root=cfg.project_root,
    )
    install_signal_handlers(get_global_token(), graceful_sec=cfg.app.graceful_shutdown_sec)
    cfg.frontend.mode = "gui"
    app = BaodouApp(cfg, mock=mock)
    app.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="baodou-ui", description="baodou GUI (Phase H)")
    p.add_argument("--config", default=None)
    p.add_argument("--live", action="store_true", help="Use real capture/vision (still dry_run)")
    p.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)
    return run_app(args.config, mock=not args.live, log_level=args.log_level)


if __name__ == "__main__":
    raise SystemExit(main())
