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


@pytest.fixture
def audio_recorder():
    return whisper_voice.AudioRecorder(config=whisper_voice.config)


@pytest.fixture
def transcriber():
    return whisper_voice.Transcriber(client=MagicMock(), config=whisper_voice.config)


@pytest.fixture
def text_output():
    return whisper_voice.TextOutput()


@pytest.fixture
def controller(audio_recorder, transcriber, text_output):
    return whisper_voice.HotkeyController(audio_recorder, transcriber, text_output)


# ---------------------------------------------------------------------------
# AudioRecorder
# ---------------------------------------------------------------------------

class TestAudioRecorder:
    def test_appends_chunk_when_recording(self, audio_recorder):
        audio_recorder.recording = True
        chunk = np.zeros((1024, 1), dtype="int16")
        audio_recorder._audio_callback(chunk, 1024, None, None)
        assert len(audio_recorder.audio_data) == 1
        np.testing.assert_array_equal(audio_recorder.audio_data[0], chunk)

    def test_ignores_chunk_when_not_recording(self, audio_recorder):
        audio_recorder.recording = False
        chunk = np.zeros((1024, 1), dtype="int16")
        audio_recorder._audio_callback(chunk, 1024, None, None)
        assert audio_recorder.audio_data == []

    def test_stop_returns_empty_array_when_no_audio(self, audio_recorder):
        audio_recorder.audio_data = []
        result = audio_recorder.stop()
        assert result.size == 0

    def test_stop_concatenates_chunks(self, audio_recorder):
        chunk = np.zeros((1024, 1), dtype="int16")
        audio_recorder.audio_data = [chunk, chunk]
        result = audio_recorder.stop()
        assert result.shape[0] == 2048


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------

class TestTranscriber:
    def test_transcribes_and_returns_stripped_text(self, transcriber):
        chunk = np.zeros((1024, 1), dtype="int16")
        audio = np.concatenate([chunk], axis=0)
        mock_result = MagicMock()
        mock_result.text = "  bonjour  "
        transcriber.client.audio.transcriptions.create.return_value = mock_result

        result = transcriber.transcribe(audio)

        assert result == "bonjour"
        transcriber.client.audio.transcriptions.create.assert_called_once()

    def test_raises_on_api_failure(self, transcriber):
        chunk = np.zeros((1024, 1), dtype="int16")
        audio = np.concatenate([chunk], axis=0)
        transcriber.client.audio.transcriptions.create.side_effect = Exception("API down")

        with pytest.raises(Exception, match="API down"):
            transcriber.transcribe(audio)


# ---------------------------------------------------------------------------
# TextOutput
# ---------------------------------------------------------------------------

class TestTextOutput:
    def test_copies_and_pastes(self, text_output):
        with patch("pyperclip.copy") as mock_copy, \
             patch("pyautogui.hotkey") as mock_hotkey:
            text_output.paste("bonjour")
        mock_copy.assert_called_once_with("bonjour")
        mock_hotkey.assert_called_once_with("ctrl", "v")


# ---------------------------------------------------------------------------
# HotkeyController — toggle
# ---------------------------------------------------------------------------

class TestHotkeyController:
    def test_starts_recording_when_idle(self, controller):
        controller.audio_recorder.recording = False
        with patch.object(controller.audio_recorder, "start") as mock_start:
            controller.toggle()
        mock_start.assert_called_once()

    def test_spawns_transcription_thread_when_recording(self, controller):
        controller.audio_recorder.recording = True
        with patch("threading.Thread") as mock_thread_cls:
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread
            controller.toggle()
        mock_thread_cls.assert_called_once_with(
            target=controller._run_transcription_pipeline, daemon=True
        )
        mock_thread.start.assert_called_once()


# ---------------------------------------------------------------------------
# HotkeyController — pipeline
# ---------------------------------------------------------------------------

class TestHotkeyControllerPipeline:
    def test_warns_when_no_audio(self, controller, capsys):
        with patch.object(controller.audio_recorder, "stop",
                          return_value=np.array([], dtype=np.int16)):
            controller._run_transcription_pipeline()
        assert "No audio captured." in capsys.readouterr().out

    def test_transcribes_and_pastes(self, controller):
        chunk = np.zeros((1024, 1), dtype="int16")
        audio = np.concatenate([chunk], axis=0)
        with patch.object(controller.audio_recorder, "stop", return_value=audio), \
             patch.object(controller.transcriber, "transcribe", return_value="bonjour"), \
             patch.object(controller.text_output, "paste") as mock_paste:
            controller._run_transcription_pipeline()
        mock_paste.assert_called_once_with("bonjour")

    def test_prints_error_on_api_failure(self, controller, capsys):
        chunk = np.zeros((1024, 1), dtype="int16")
        audio = np.concatenate([chunk], axis=0)
        with patch.object(controller.audio_recorder, "stop", return_value=audio), \
             patch.object(controller.transcriber, "transcribe",
                          side_effect=Exception("API down")):
            controller._run_transcription_pipeline()
        assert "Error:" in capsys.readouterr().out
