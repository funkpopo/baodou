"""CLI smoke tests."""

from __future__ import annotations

from frontend.cli import main


def test_cli_version(capsys) -> None:
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "protocol" in out
    assert "0.1.0" in out or "1.0.0" in out


def test_cli_config_show(capsys) -> None:
    assert main(["config-show"]) == 0
    out = capsys.readouterr().out
    assert "inference" in out
    assert "SYCL0" in out


def test_cli_demo(capsys, tmp_path) -> None:
    out_path = tmp_path / "demo.json"
    rc = main(
        ["demo", "--goal", "描述当前屏幕", "--json-out", str(out_path), "--log-level", "WARNING"]
    )
    assert rc == 0
    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "trace_id" in text
    assert "events" in text
