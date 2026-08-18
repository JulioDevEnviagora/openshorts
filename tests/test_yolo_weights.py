"""YOLO weights must load from the image, not a root-owned /app cwd."""
from pathlib import Path

import yolo_weights

ROOT = Path(__file__).resolve().parents[1]


def test_env_override_when_file_exists(tmp_path, monkeypatch):
    weights = tmp_path / "custom.pt"
    weights.write_bytes(b"x")
    monkeypatch.setenv("YOLO_WEIGHTS", str(weights))
    assert yolo_weights.yolo_weights_path() == str(weights)


def test_missing_env_path_is_ignored(monkeypatch):
    monkeypatch.setenv("YOLO_WEIGHTS", "/definitely/missing/yolov8n.pt")
    monkeypatch.setattr(yolo_weights, "BAKED_YOLO_WEIGHTS", "/also/missing.pt")
    assert yolo_weights.yolo_weights_path() == "yolov8n.pt"


def test_baked_path_when_present(tmp_path, monkeypatch):
    baked = tmp_path / "yolov8n.pt"
    baked.write_bytes(b"x")
    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    monkeypatch.setattr(yolo_weights, "BAKED_YOLO_WEIGHTS", str(baked))
    assert yolo_weights.yolo_weights_path() == str(baked)


def test_default_relative_name_when_baked_missing(monkeypatch):
    monkeypatch.delenv("YOLO_WEIGHTS", raising=False)
    monkeypatch.setattr(yolo_weights, "BAKED_YOLO_WEIGHTS", "/also/missing.pt")
    assert yolo_weights.yolo_weights_path() == "yolov8n.pt"


def test_selfhost_compose_does_not_bind_mount_checkout():
    text = (ROOT / "docker-compose.yml").read_text()
    assert "- .:/app" not in text
    assert "YOLO_WEIGHTS=/opt/models/yolov8n.pt" in text
    assert "MPLCONFIGDIR=/tmp/matplotlib" in text
    assert "uploads_data:/app/uploads" in text


def test_dev_compose_keeps_bind_mount_for_hmr():
    text = (ROOT / "docker-compose.dev.yml").read_text()
    assert "- .:/app" in text


def test_dockerfile_bakes_weights_outside_app():
    text = (ROOT / "Dockerfile").read_text()
    assert "YOLO_WEIGHTS=/opt/models/yolov8n.pt" in text
    assert "MPLCONFIGDIR=/tmp/matplotlib" in text
    assert "mv yolov8n.pt /opt/models/yolov8n.pt" in text


def test_main_loads_resolved_weights_path():
    text = (ROOT / "main.py").read_text()
    assert "YOLO(yolo_weights_path())" in text
    assert "YOLO('yolov8n.pt')" not in text
