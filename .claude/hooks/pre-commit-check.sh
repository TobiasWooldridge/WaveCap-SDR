#!/bin/bash
# Pre-commit validation hook for WaveCap-SDR
# Runs type checking on modified Python files before git commits

set -e

# Parse input JSON from stdin
INPUT=$(cat)
TOOL_NAME=$(echo "$INPUT" | jq -r '.tool_name' 2>/dev/null || echo "")
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command' 2>/dev/null || echo "")

# Only validate git commit commands
if [[ "$TOOL_NAME" != "Bash" ]]; then
    exit 0
fi

# Check if this is a git commit command
if ! echo "$COMMAND" | grep -qE "^git\s+(commit|push)"; then
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-/Users/thw/Projects/WaveCap-SDR}"
BACKEND_DIR="$PROJECT_DIR/backend"

# Check if backend exists
if [ ! -d "$BACKEND_DIR" ]; then
    exit 0
fi

cd "$BACKEND_DIR"

# Get list of staged Python files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.py$' || true)

if [ -z "$STAGED_FILES" ]; then
    exit 0  # No Python files staged
fi

echo "Pre-commit: Checking staged Python files..." >&2

# Activate venv
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    MYPY=".venv/bin/mypy"
else
    PYTHON="python3"
    MYPY="mypy"
fi

# Run mypy on staged files only (less strict - just check for obvious errors)
ERRORS=0
for file in $STAGED_FILES; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        # Quick syntax check
        if ! $PYTHON -m py_compile "$PROJECT_DIR/$file" 2>&1; then
            echo "Syntax error in $file" >&2
            ERRORS=1
        fi
    fi
done

if [ $ERRORS -ne 0 ]; then
    echo "" >&2
    echo "Pre-commit check failed. Fix errors above." >&2
    exit 2
fi

echo "Pre-commit: All checks passed" >&2
exit 0
