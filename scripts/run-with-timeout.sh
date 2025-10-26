#!/usr/bin/env bash
set -euo pipefail

# Generic timeout wrapper to prevent long-running or stalled commands.
# - Defaults to 120 seconds; override via MAX_SECONDS or --seconds/-s
# - Uses coreutils `timeout` when available, else a Python fallback.
# - Ensures processes are terminated (TERM then KILL) and returns 124 on timeout.
#
# Usage examples:
#   scripts/run-with-timeout.sh -- echo "hello"
#   scripts/run-with-timeout.sh --seconds 30 -- pytest -q
#   MAX_SECONDS=15 scripts/run-with-timeout.sh -- python -m wavecapsdr.harness ...

SECONDS_DEFAULT=${MAX_SECONDS:-120}
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

# Optional separator `--` before the command
if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "usage: run-with-timeout.sh [--seconds N] -- <command> [args...]" >&2
  exit 2
fi

if command -v timeout >/dev/null 2>&1; then
  set +e
  timeout --signal TERM -k 2s "${SECONDS_VALUE}s" "$@"
  status=$?
  set -e
  if [[ $status -eq 124 ]]; then
    echo "Command timed out after ${SECONDS_VALUE}s: $*" >&2
  fi
  exit $status
fi

# Fallback to Python implementation if `timeout` is unavailable
python3 - "$SECONDS_VALUE" -- "$@" <<'PY'
import os, sys, subprocess, signal

def main():
    if len(sys.argv) < 3 or sys.argv[2] != "--":
        print("usage: <timeout_sec> -- <program> [args...]", file=sys.stderr)
        return 2
    try:
        timeout_sec = float(sys.argv[1])
    except ValueError:
        print("invalid timeout", file=sys.stderr)
        return 2
    prog = sys.argv[3]
    args = sys.argv[4:]
    try:
        # Start in its own process group so we can kill the whole tree if needed
        preexec = os.setsid if hasattr(os, 'setsid') else None
        p = subprocess.Popen([prog, *args], preexec_fn=preexec)
        try:
            return p.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                # TERM the group first
                if preexec is not None:
                    os.killpg(p.pid, signal.SIGTERM)
                else:
                    p.terminate()
            except Exception:
                pass
            try:
                p.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    if preexec is not None:
                        os.killpg(p.pid, signal.SIGKILL)
                    else:
                        p.kill()
                except Exception:
                    pass
            print(f"Command timed out after {timeout_sec}s: {prog}", file=sys.stderr)
            return 124
    except FileNotFoundError:
        print(f"{prog} not found", file=sys.stderr)
        return 127

if __name__ == "__main__":
    sys.exit(main())
PY
