#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: SoapySDRUtil --watch with a hard timeout.
# Usage:
#   scripts/soapy-watch.sh                     # watch default (10s)
#   scripts/soapy-watch.sh driver=sdrplay
#   scripts/soapy-watch.sh --seconds 6 driver=rtlsdr

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
  exec "$UTIL" "${SECONDS_ARG[@]}" --watch="${ARGS}"
else
  exec "$UTIL" "${SECONDS_ARG[@]}" --watch
fi

