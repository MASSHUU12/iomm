#!/usr/bin/env bash
set -euo pipefail

# (Re)create virtualenv
VENV_DIR=".venv"
python3 -m venv "$VENV_DIR"
source "${VENV_DIR}/bin/activate"

# Install dependencies
pip install --upgrade pip
pip install \
  pytest \
  pytest-benchmark

echo "Installed packages:"
pip freeze
