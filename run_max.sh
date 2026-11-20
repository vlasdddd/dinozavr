#!/usr/bin/env bash
# Максимальный забег DQN-динозавра (Linux / macOS).
set -e
cd "$(dirname "$0")"

if [ -x "./venv/bin/python" ]; then
    PY="./venv/bin/python"
else
    PY="python3"
fi

exec "$PY" -u run_max.py "$@"
