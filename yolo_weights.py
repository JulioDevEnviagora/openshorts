"""Resolve YOLOv8 weights without writing into a read-only /app cwd.

The image bakes `yolov8n.pt` at BAKED_YOLO_WEIGHTS (outside /app). A Dokploy
checkout bind-mounted over /app is root-owned and hides a cwd copy, so YOLO
must never fall back to downloading `yolov8n.pt` into /app.
"""
import os

BAKED_YOLO_WEIGHTS = "/opt/models/yolov8n.pt"


def yolo_weights_path() -> str:
    """Return an existing weights path, or the local-dev filename.

    Prefer YOLO_WEIGHTS when that file exists. Otherwise the baked image
    path. The bare `yolov8n.pt` name is last — Ultralytics then looks in
    cwd and will download there, which is fine for a writable local tree
    and fatal on a root-owned Dokploy mount.
    """
    configured = (os.environ.get("YOLO_WEIGHTS") or "").strip()
    if configured and os.path.isfile(configured):
        return configured
    if os.path.isfile(BAKED_YOLO_WEIGHTS):
        return BAKED_YOLO_WEIGHTS
    return "yolov8n.pt"
