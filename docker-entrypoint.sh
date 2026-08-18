#!/bin/sh
# Ensure the working directories exist and are writable by appuser, then drop
# root. Named volumes and Dokploy bind-mounts are often created as root:root;
# uvicorn runs as appuser and os.makedirs('uploads') would otherwise EACCES.
set -e

upload_dir="${UPLOAD_DIR:-/app/uploads}"
output_dir="${OUTPUT_DIR:-/app/output}"

mkdir -p "$upload_dir" "$output_dir"

if [ "$(id -u)" = "0" ]; then
  chown appuser:appuser "$upload_dir" "$output_dir"
  exec gosu appuser "$@"
fi

exec "$@"
