#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper: SoapySDRUtil --probe with a hard timeout.
# Usage:
#   scripts/soapy-probe.sh                     # probe default
#   scripts/soapy-probe.sh driver=sdrplay      # probe specific driver
#   scripts/soapy-probe.sh driver=rtlsdr serial=00000001
#   MAX_SECONDS=8 scripts/soapy-probe.sh driver=rtlsdr
#   scripts/soapy-probe.sh --seconds 6 driver=sdrplay

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
  exec "$UTIL" "${SECONDS_ARG[@]}" --probe="${ARGS}"
else
  exec "$UTIL" "${SECONDS_ARG[@]}" --probe
fi

