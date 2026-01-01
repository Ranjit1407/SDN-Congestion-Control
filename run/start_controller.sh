#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ryu-venv-3.9/bin/activate
python -m ryu.cmd.manager controller/Controller_UnifiedController.py

