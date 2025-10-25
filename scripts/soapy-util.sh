#!/usr/bin/env bash
set -euo pipefail

# Wrapper for SoapySDRUtil that enforces a hard timeout.
# - Defaults to 10 seconds (override with env MAX_SECONDS or --seconds/-s)
# - Preserves exit codes when possible; returns 124 on timeout (like coreutils timeout)
# - Example:
#     scripts/soapy-util.sh --find
#     scripts/soapy-util.sh --seconds 8 --probe="driver=sdrplay"

SECONDS_DEFAULT=${MAX_SECONDS:-10}
SECONDS_VALUE=$SECONDS_DEFAULT

if [[ "${1:-}" == "--seconds" || "${1:-}" == "-s" ]]; then
  shift
  if [[ $# -lt 1 ]]; then
    echo "--seconds requires a numeric argument" >&2
    exit 2
  fi
  SECONDS_VALUE="$1"
  shift
fi

SOAPY_BIN=${SOAPYSDRUTIL:-SoapySDRUtil}
if ! command -v "$SOAPY_BIN" >/dev/null 2>&1; then
  echo "SoapySDRUtil not found in PATH (looked for '$SOAPY_BIN'). Install SoapySDR utilities." >&2
  exit 127
fi

if command -v timeout >/dev/null 2>&1; then
  # Use coreutils timeout with a gentle TERM and hard KILL after 2s
  set +e
  timeout --preserve-status --signal TERM -k 2s "${SECONDS_VALUE}s" "$SOAPY_BIN" "$@"
  status=$?
  set -e
  if [[ $status -eq 124 ]]; then
    echo "SoapySDRUtil timed out after ${SECONDS_VALUE}s" >&2
  fi
  exit $status
fi

# Fallback to Python timeout if coreutils timeout is unavailable
python3 - "$SECONDS_VALUE" "$SOAPY_BIN" "$@" <<'PY'
import os, sys, subprocess, signal

def main():
    if len(sys.argv) < 3:
        print("usage: <timeout_sec> <program> [args...]", file=sys.stderr)
        return 2
    try:
        timeout_sec = float(sys.argv[1])
    except ValueError:
        print("invalid timeout", file=sys.stderr)
        return 2
    prog = sys.argv[2]
    args = sys.argv[3:]
    try:
        p = subprocess.Popen([prog, *args])
        try:
            return p.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
            print(f"{prog} timed out after {timeout_sec}s", file=sys.stderr)
            return 124
    except FileNotFoundError:
        print(f"{prog} not found", file=sys.stderr)
        return 127

if __name__ == "__main__":
    sys.exit(main())
PY

