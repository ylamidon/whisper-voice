# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A single-file Python script that binds a global hotkey (`Ctrl+Alt+Space`) to toggle audio recording. On the second press, it transcribes the audio using OpenAI's Whisper API and pastes the result into the active window via the clipboard.

## Setup & Running

Install dependencies:
```
uv sync
```

Configure API key via `.env` file:
```
OPENAI_API_KEY=sk-...
```

Run the script (requires admin/elevated privileges on Windows for global hotkey capture):
```
uv run python whisper_voice.py
```

Stop with `Ctrl+C` in the terminal.

Run tests:
```
uv run pytest
```

## Key Configuration

At the top of [whisper_voice.py](whisper_voice.py):

| Variable | Default | Purpose |
|----------|---------|---------|
| `HOTKEY` | `ctrl+alt+space` | Toggle recording |
| `LANGUAGE` | `fr` | Whisper transcription language |
| `SAMPLERATE` | `16000` | Audio sample rate (Hz) |
| `MODEL` | `whisper-1` | OpenAI Whisper model |

## Architecture

The script is entirely self-contained in `whisper_voice.py`. The flow:

1. `main()` registers the hotkey and blocks on `keyboard.wait("ctrl+c")`
2. `toggle()` switches between `start_recording()` and `stop_and_transcribe()`
3. `start_recording()` opens a `sounddevice.InputStream` with `audio_callback` collecting chunks into `audio_data[]`
4. `stop_and_transcribe()` concatenates chunks with numpy, writes a temp `.wav`, calls the OpenAI Whisper API, copies the result to clipboard, and pastes with `pyautogui`

Thread safety: `audio_callback` and `recording` flag are guarded by a `threading.Lock`. Transcription runs in a daemon thread to avoid blocking the hotkey listener.
