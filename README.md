# WaveCap‑SDR

A standalone server that encapsulates SDR device control, capture, and demodulation for the WaveCap ecosystem. It exposes a simple network API so other services can list devices, start/stop tuned captures, create multiple demod channels from a single device stream, stream IQ or audio, and record to disk.

- Spec: see `SPEC.md` (authoritative) for scope, APIs, and milestones.
- Contribution & workflow: see `AGENTS.md` for coding principles, testing expectations, and repo conventions.

## Status
Pre‑alpha scaffold. Initial milestones: device enumeration, IQ streaming, and multi‑channel demod within a single capture (e.g., multiple VHF marine channels from one tuner).

## Getting Started
This repository is server‑focused. The exact implementation language and runtime scaffolding will be introduced with the first milestone. Until then:

- Review `SPEC.md` for the planned API and components.
- If you’re preparing to contribute, follow `AGENTS.md` conventions (strong typing, tests, and docs updates).

When the backend is bootstrapped, this section will include concrete setup commands, environment configuration, and how to run the test suite.

## Relation to WaveCap
- WaveCap (control/UI) lives in `~/speaker/WaveCap` (also symlinked as `~/speaker/smart-speaker`). WaveCap‑SDR provides the radio server component. Together they form one product; this repo intentionally contains no frontend.

## Repository Layout (planned)
- `backend/` — server code and tests
- `docs/` — documentation (`docs/configuration.md` covers all runtime options)
- `scripts/` — helper scripts for development and CI

## License
TBD
