"""Working dirs must be creatable under Dokploy's root-owned bind-mount.

app.py used to hardcode `uploads`/`output` under cwd. The image runs as
appuser; Dokploy bind-mounts the clone over /app as root, so
os.makedirs('uploads') raised PermissionError and uvicorn never listened.
"""
import os
import subprocess
from pathlib import Path

import pytest

app = pytest.importorskip("app")

ROOT = Path(__file__).resolve().parents[1]


def test_data_dir_honors_env(tmp_path, monkeypatch):
    target = tmp_path / "custom-uploads"
    monkeypatch.setenv("UPLOAD_DIR", str(target))
    path = app._data_dir("UPLOAD_DIR", "uploads")
    assert path == str(target)
    assert target.is_dir()


def test_data_dir_defaults_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("UPLOAD_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    path = app._data_dir("UPLOAD_DIR", "uploads")
    assert path == "uploads"
    assert (tmp_path / "uploads").is_dir()


def test_entrypoint_creates_dirs_without_root(tmp_path):
    """Non-root path of docker-entrypoint.sh (no gosu) still mkdir -p's."""
    upload = tmp_path / "uploads"
    output = tmp_path / "output"
    script = ROOT / "docker-entrypoint.sh"
    env = os.environ.copy()
    env["UPLOAD_DIR"] = str(upload)
    env["OUTPUT_DIR"] = str(output)
    subprocess.run(
        ["/bin/sh", str(script), "true"],
        check=True,
        env=env,
        cwd=tmp_path,
    )
    assert upload.is_dir()
    assert output.is_dir()
