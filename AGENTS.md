# AGENTS.md — WaveCap‑SDR

This file gives concise guidance to humans and coding agents working in this repository.
Scope: applies to the entire repo.

## Coding Principles
- Prefer strong typing throughout. Enable strict type checks:
  - Python: configure mypy in strict mode and keep the code type‑clean.
  - TypeScript: enable `"strict": true` if TS is used anywhere.
- Keep implementations simple and purposeful; avoid unnecessary abstractions.
- Encapsulate primitive components/utilities so callers don’t repeat defaults.
- Favor explicit, well‑named functions over deep inheritance.

## Testing Philosophy
- Write tests for real code paths. Minimize mocking; only mock heavy or external edges (e.g., SDR hardware/driver bindings). Provide a thin driver abstraction so logic can be exercised without hardware attached.
- Prefer integration tests for capture + demod pipelines using recorded IQ fixtures where hardware is unavailable.
- Add tests alongside features; do not defer.

## Documentation Expectations
- Treat `docs/api-spec.md` as the authoritative product spec. Update it with any functional change, even small.
- Document every meaningful configuration option in `docs/configuration.md` (create if missing). Keep options discoverable and auditable.
- In user‑facing docs (README, guides), focus on workflows and outcomes. Avoid naming frameworks/libraries unless strictly needed for setup or troubleshooting.
- Define diagrams using Mermaid blocks instead of exporting static PNGs or similar assets.

## Workflow Notes
- Follow repository-specific instructions in nested `AGENTS.md` files if present; their scope applies to their directory and descendants.
- Always run the linters/tests relevant to the areas you touch before committing changes.
- Run required checks before publishing a change:
  - Backend tests: `cd backend && source .venv/bin/activate && PYTHONPATH=. pytest tests/`
  - Backend types: `cd backend && source .venv/bin/activate && mypy wavecapsdr`
  - Frontend (no unit tests): `cd frontend && npm run type-check && npm run lint && npm run build`
- Keep the working tree clean with meaningful commits.
- Before publishing or opening a PR, sync with remote if you have access.
- Always validate that the code compiles/builds before completion.
- Include at least one screenshot in every pull request description.
  - When the change affects UI or a demo surface, include a browser capture of the scenario being exercised.
  - For backend‑only changes, include a relevant capture (e.g., metrics dashboard, CLI run, or API trace) that demonstrates the behavior.
  - If `./start-screenshot.sh` exists in the repo, prefer it to prepare captures; otherwise capture manually.
- Push back on requests that bloat complexity or drift from purpose.

## Project Conventions
- Directory layout (initial expectation):
  - `backend/` — server code, typed, with tests under `backend/tests/`.
  - `frontend/` — web UI sources (built output copied to `backend/wavecapsdr/static/`).
  - `docs/` — user/admin docs; include `configuration.md`.
  - `scripts/` — helper scripts for development and CI.
  - Note: WaveCap app lives alongside at `~/speaker/WaveCap`. Keep integration flows documented there; reference this repo’s SPEC for SDR API.
- Logging and metrics are part of the surface area; keep them consistent and documented.
  - Model runtime around: Devices → Captures → Channels. Unit and integration tests should reflect that layering.

## Agent Notes (for automated assistants)
- Respect this file’s scope and instructions when editing code.
- Use plans for multi‑step or ambiguous work; keep steps short and sequential.
- Prefer surgical, minimal diffs. Avoid renames or broad refactors unless explicitly requested.
- Do not create commits or branches unless the user asks. Prepare changes as patches when collaborating in tooling.

## Running the Server
For development and testing, use the startup scripts from the repository root:

- Linux/macOS: `./start-app.sh`
- Windows PowerShell: `.\start-app.ps1`

These scripts handle virtual environment setup, dependency installation, and start the server with sensible defaults. Environment variables can customize behavior:
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8087`)
- `DRIVER` (default: `soapy`)
- `DEVICE_ARGS` (optional, e.g., `"driver=rtlsdr"`)
- `CONFIG` (optional, path to config file)
- Use `./restart-sdrplay.sh` whenever the SDRplay systemd service needs a manual restart (script wraps `sudo systemctl restart sdrplay`). `start-app.sh` automatically attempts a non-interactive restart via this helper, but if sudo prompts for a password you'll need to run the helper yourself before starting the app.
- When radios misbehave (especially SDRplay enumeration or streaming issues), run `scripts/fix-sdrplay.sh` to power-cycle and reinitialize the device.

## Test Harness (for agentic tools)
Use the built‑in harness to spin up a local server, create a capture, add channels, stream audio, and validate levels. It works with the fake driver (offline) and SoapySDR (RTL‑SDR, SDRplay RSPdx‑r2).

- Quick start (offline):
  - `cd ~/speaker/WaveCap-SDR/backend`
  - `python -m venv .venv && . .venv/bin/activate`
  - `pip install fastapi uvicorn httpx websockets pyyaml numpy scipy`
  - `PYTHONPATH=. python -m wavecapsdr.harness --start-server --driver fake --preset marine --duration 3 --out ./harness_out`
  - Output JSON includes per‑channel RMS/peak; WAV files saved under `harness_out/`.

- Real hardware (RTL‑SDR or RSPdx‑r2 via SoapySDR):
  - Ensure SoapySDR and the appropriate module (e.g., `SoapyRTLSDR` or `SoapySDRPlay3`) are installed on the host.
  - KEXP preset (90.3 MHz, WBFM):
    - `bash scripts/harness-kexp.sh`
      - Defaults to `backend/config/wavecapsdr.local.yaml` if present (overlays `backend/config/wavecapsdr.yaml`), which contains `presets.kexp`.
      - Optional device selection:
      - RTL‑SDR: `DEVICE_ARGS="driver=rtlsdr" bash scripts/harness-kexp.sh`
      - RSPdx‑r2: `DEVICE_ARGS="driver=sdrplay" bash scripts/harness-kexp.sh`
  - Marine multi‑channel example:
    - `bash scripts/harness-marine.sh`

- Custom tuning:
  - `PYTHONPATH=. python -m wavecapsdr.harness --start-server --driver fake --center-hz 1.568e8 --sample-rate 1200000 --offset 0 --offset 50000 --duration 4`
  - Offsets are in Hz relative to `center-hz`. Repeat `--offset` for multiple channels.

- What it validates:
  - Starts server (if requested) and creates a capture with given center/sample rate.
  - Adds channels (WBFM), starts them, streams audio over WebSocket.
  - Measures RMS/peak and writes WAVs for inspection. Exits non‑zero if audio level is too low.

CI/automation tips
- Scripts return non‑zero if audio not detected above thresholds; agents can treat this as a failed check.
- WAVs are stored under `backend/harness_out/` for post‑run inspection.

## Timeout wrapper (generic)
To avoid indefinite stalls when running commands that may hang or take too long, use the generic timeout wrapper. This is required for agentic/CI runs; prefer it during local development for long‑running tasks.

- Script: `scripts/run-with-timeout.sh`
- Defaults: 120s timeout; returns exit code 124 on timeout.
- Uses coreutils `timeout` when available, otherwise a Python fallback.
- Examples:
  - `scripts/run-with-timeout.sh --seconds 40 -- bash scripts/harness-kexp.sh`
  - `scripts/run-with-timeout.sh --seconds 60 -- env PYTHONPATH=. pytest -q`
  - `scripts/run-with-timeout.sh --seconds 30 -- bash -lc 'cd backend && . .venv/bin/activate && python -m wavecapsdr.harness --start-server --driver fake --preset kexp --duration 3'`
- Notes:
  - Continue to use the Soapy wrappers (`scripts/soapy-*.sh`) for SoapySDRUtil; the generic timeout is for everything else (tests, harness, ad‑hoc commands).
  - On timeout (124), capture logs and artifacts and retry with more verbosity if needed.

## SoapySDRUtil wrapper
When probing hardware or testing rates with SoapySDRUtil, you MUST use the timeout wrapper so commands never hang indefinitely. Do not call `SoapySDRUtil` directly.

- Run with default 10s timeout:
  - `scripts/soapy-find.sh`
  - `scripts/soapy-probe.sh driver=sdrplay`
- Customize timeout:
  - `scripts/soapy-watch.sh --seconds 8 driver=rtlsdr` (auto-terminates after 8s)
  - or `MAX_SECONDS=5 scripts/soapy-find.sh`
- Notes:
  - Uses coreutils `timeout` when available; otherwise falls back to a Python implementation.
  - Returns exit code 124 on timeout (same as `timeout`).

- Integration with WaveCap (sibling repo):
  - WaveCap lives at `~/speaker/WaveCap` and orchestrates scenarios. Use the harness here to verify radio configs independently before wiring flows in WaveCap.
