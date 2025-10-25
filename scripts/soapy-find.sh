#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: SoapySDRUtil --find with a hard timeout.
# Usage:
#   scripts/soapy-find.sh                      # find with default 10s timeout
#   MAX_SECONDS=5 scripts/soapy-find.sh        # 5s timeout via env
#   scripts/soapy-find.sh --seconds 8          # 8s timeout via flag
#   scripts/soapy-find.sh driver=rtlsdr        # limit find to a specific driver

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL="$SCRIPT_DIR/soapy-util.sh"

if [[ $# -gt 0 && ( "$1" == "--seconds" || "$1" == "-s" ) ]]; then
  SECONDS_ARG=("$1" "$2")
  shift 2
else
  SECONDS_ARG=()
fi

if [[ $# -gt 0 ]]; then
  ARGS=$(printf ",%s" "$@")
  ARGS="${ARGS:1}"
  exec "$UTIL" "${SECONDS_ARG[@]}" --find="${ARGS}"
else
  exec "$UTIL" "${SECONDS_ARG[@]}" --find
fi

