import os
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Set env var and mock OpenAI before the module is imported so the
# module-level guards don't call sys.exit or make real network calls.
os.environ["OPENAI_API_KEY"] = "test-key"

with patch("openai.OpenAI"):
    import whisper_voice


@pytest.fixture(autouse=True)
def reset_state():
    """Restore module globals to a clean state around every test."""
    whisper_voice.recording = False
    whisper_voice.audio_data = []
    whisper_voice.stream = None
    yield
    whisper_voice.recording = False
    whisper_voice.audio_data = []
    whisper_voice.stream = None


# ---------------------------------------------------------------------------
# audio_callback
# ---------------------------------------------------------------------------

class TestAudioCallback:
    def test_appends_chunk_when_recording(self):
        whisper_voice.recording = True
        chunk = np.zeros((1024, 1), dtype="int16")
        whisper_voice.audio_callback(chunk, 1024, None, None)
        assert len(whisper_voice.audio_data) == 1
        np.testing.assert_array_equal(whisper_voice.audio_data[0], chunk)

    def test_ignores_chunk_when_not_recording(self):
        whisper_voice.recording = False
        chunk = np.zeros((1024, 1), dtype="int16")
        whisper_voice.audio_callback(chunk, 1024, None, None)
        assert whisper_voice.audio_data == []


# ---------------------------------------------------------------------------
# toggle
# ---------------------------------------------------------------------------

class TestToggle:
    def test_starts_recording_when_idle(self):
        whisper_voice.recording = False
        with patch.object(whisper_voice, "start_recording") as mock_start:
            whisper_voice.toggle()
        mock_start.assert_called_once()

    def test_spawns_transcribe_thread_when_recording(self):
        whisper_voice.recording = True
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            whisper_voice.toggle()
        mock_thread_cls.assert_called_once_with(
            target=whisper_voice.stop_and_transcribe, daemon=True
        )
        mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# stop_and_transcribe
# ---------------------------------------------------------------------------

class TestStopAndTranscribe:
    def test_warns_when_no_audio(self, capsys):
        whisper_voice.audio_data = []
        whisper_voice.stop_and_transcribe()
        assert "No audio captured." in capsys.readouterr().out

    def test_transcribes_and_pastes(self, capsys):
        chunk = np.zeros((1024, 1), dtype="int16")
        whisper_voice.audio_data = [chunk]
        mock_result = MagicMock()
        mock_result.text = "  bonjour  "
        whisper_voice.client.audio.transcriptions.create.return_value = mock_result

        with patch("pyperclip.copy") as mock_copy, \
             patch("pyautogui.hotkey") as mock_hotkey:
            whisper_voice.stop_and_transcribe()

        mock_copy.assert_called_once_with("bonjour")
        mock_hotkey.assert_called_once_with("ctrl", "v")
        assert "bonjour" in capsys.readouterr().out

    def test_prints_error_on_api_failure(self, capsys):
        chunk = np.zeros((1024, 1), dtype="int16")
        whisper_voice.audio_data = [chunk]
        whisper_voice.client.audio.transcriptions.create.side_effect = Exception("API down")

        with patch("pyperclip.copy"), patch("pyautogui.hotkey"):
            whisper_voice.stop_and_transcribe()

        assert "Error:" in capsys.readouterr().out
