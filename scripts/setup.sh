#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${1:-.venv}"

python3 -m venv "${VENV_PATH}"
if ! "${VENV_PATH}/bin/pip" install --upgrade pip setuptools wheel; then
  echo "[WARN] Could not upgrade pip/setuptools/wheel. Continuing with existing tooling."
fi

if ! "${VENV_PATH}/bin/pip" install -r requirements-dev.txt; then
  echo "[ERROR] Dependency installation failed. Check network access and retry."
  exit 1
fi

echo "Setup complete."
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "Run tests with: ${VENV_PATH}/bin/pytest -q"
