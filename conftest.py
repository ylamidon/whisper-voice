from unittest.mock import MagicMock

import pytest

import whisper_voice


@pytest.fixture
def config():
    return whisper_voice.Config()


@pytest.fixture
def audio_recorder(config):
    return whisper_voice.AudioRecorder(config=config)


@pytest.fixture
def transcriber(config):
    return whisper_voice.Transcriber(client=MagicMock(), config=config)


@pytest.fixture
def text_output():
    return whisper_voice.TextOutput()


@pytest.fixture
def controller(audio_recorder, transcriber, text_output):
    return whisper_voice.HotkeyController(audio_recorder, transcriber, text_output)
