# Audio Quality Checker Skill

A Claude Code skill for analyzing WaveCap-SDR audio streams to assess tuning quality.

## Quick Start

1. Make sure the WaveCap-SDR server is running
2. Activate the skill by asking Claude to check audio quality
3. Claude will use the `analyze_audio_stream.py` script to capture and analyze audio

## Manual Usage

```bash
# Analyze default channel (ch1) on default port (8087)
PYTHONPATH=backend backend/.venv/bin/python .claude/skills/audio-quality-checker/analyze_audio_stream.py

# Analyze specific channel and port
PYTHONPATH=backend backend/.venv/bin/python .claude/skills/audio-quality-checker/analyze_audio_stream.py \
  --port 8088 --channel ch2 --duration 5

# All options
PYTHONPATH=backend backend/.venv/bin/python .claude/skills/audio-quality-checker/analyze_audio_stream.py \
  --host 127.0.0.1 \
  --port 8087 \
  --channel ch1 \
  --duration 3 \
  --format pcm16
```

## What It Detects

- **Silence**: No signal (broken/stopped channel)
- **Noise**: Random signal (poor tuning, no carrier)
- **Good Audio**: Structured signal (well-tuned station)
- **Clipping**: Signal hitting limits (gain too high)
- **Weak Audio**: Signal present but low level

## Files

- `SKILL.md`: Claude Code skill definition
- `analyze_audio_stream.py`: Audio analysis script
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Dependencies

All required packages are already installed in `backend/.venv`:
- numpy
- scipy
- requests
