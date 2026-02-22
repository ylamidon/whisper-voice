# whisper-voice

Global hotkey voice-to-text for Windows using OpenAI Whisper. Press `Ctrl+Alt+Space` to start recording, press again to stop — the transcription is pasted directly into the active window.

## Requirements

- Windows
- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

```
uv sync
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

## Usage

Run as administrator (required for global hotkey capture):
```
uv run python whisper_voice.py
```

- `Ctrl+Alt+Space` — start recording
- `Ctrl+Alt+Space` — stop, transcribe, and paste
- `Ctrl+C` — quit

## Configuration

Edit the constants at the top of `whisper_voice.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOTKEY` | `ctrl+alt+space` | Toggle recording |
| `LANGUAGE` | `fr` | Transcription language (e.g. `en`, `fr`) |
| `SAMPLERATE` | `16000` | Audio sample rate (Hz) |
| `MODEL` | `whisper-1` | OpenAI Whisper model |
