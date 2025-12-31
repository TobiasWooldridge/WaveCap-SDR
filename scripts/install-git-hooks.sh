#!/usr/bin/env bash
set -euo pipefail

# Configure git to use the repository's managed hooks directory.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

git config core.hooksPath .githooks
echo "Git hooks configured to use: ${repo_root}/.githooks"
echo "Re-run this script after cloning to enable the managed pre-commit hook."
