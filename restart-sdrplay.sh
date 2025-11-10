#!/usr/bin/env bash
set -euo pipefail

NON_INTERACTIVE=0
if [[ "${1:-}" == "--non-interactive" ]]; then
  NON_INTERACTIVE=1
  shift || true
fi

if ! command -v systemctl >/dev/null 2>&1; then
  echo "systemctl not available; cannot restart sdrplay service" >&2
  exit 1
fi

if [[ $EUID -eq 0 ]]; then
  systemctl restart sdrplay
  exit 0
fi

if [[ $NON_INTERACTIVE -eq 1 ]]; then
  if sudo -n systemctl restart sdrplay; then
    exit 0
  fi
  echo "Warning: sudo password required to restart sdrplay; run ./restart-sdrplay.sh manually." >&2
  exit 1
fi

sudo systemctl restart sdrplay
