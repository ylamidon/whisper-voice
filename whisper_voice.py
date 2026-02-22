"""
Whisper Voice Input - Windows
==============================
Shortcut : Ctrl+Alt+Space  →  starts recording
           Ctrl+Alt+Space  →  stops, transcribes, and pastes the text

Installation:
    uv sync

Configuration:
    Put your API key in a .env file: OPENAI_API_KEY=sk-...

Flow:
    main() creates one instance of each class and wires them together via
    HotkeyController. Each hotkey press calls HotkeyController.toggle(), which
    either starts recording (AudioRecorder.start) or spawns a daemon thread that
    stops the stream, transcribes via the OpenAI Whisper API, and pastes the
    result into the active window.
"""

import os
import sys
import time
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import scipy.io.wavfile as wav
import pyperclip
import keyboard
import pyautogui
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    """Immutable application configuration.

    Attributes:
        hotkey: Global keyboard shortcut that toggles recording on/off.
        language: BCP-47 language code passed to the Whisper API.
        samplerate: Audio capture sample rate in Hz.
        model: OpenAI Whisper model identifier.
    """

    hotkey:     str = "ctrl+alt+space"
    language:   str = "fr"          # French — change to "en" if needed
    samplerate: int = 16000
    model:      str = "whisper-1"

config = Config()

# ──────────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("OPENAI_API_KEY missing. Create a .env file with OPENAI_API_KEY=sk-...")


class AudioRecorder:
    """Manages microphone capture via a sounddevice InputStream.

    Collects raw 16-bit mono PCM chunks into a list while recording is active.
    Thread safety is guaranteed by a Lock shared with HotkeyController.toggle().
    """

    def __init__(self, config: Config) -> None:
        """Initialise the recorder in an idle (not recording) state.

        Args:
            config: Application configuration supplying samplerate.
        """
        self.config     = config
        self.recording: bool = False
        self.audio_data: list[npt.NDArray[np.int16]] = []
        self.stream:    Optional[sd.InputStream] = None
        self.lock       = threading.Lock()

    def _audio_callback(
        self,
        indata: npt.NDArray[np.int16],
        _frames: int,
        _time_info: object,
        _status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback invoked for each captured audio block.

        Appends a copy of the incoming block to audio_data when recording is
        active. Copies are taken to avoid referencing the transient buffer that
        sounddevice may reuse across calls.

        Args:
            indata: Audio samples for the current block, shape (frames, channels).
            _frames: Number of frames in the block (unused).
            _time_info: Timing information from sounddevice (unused).
            _status: Stream status flags from sounddevice (unused).
        """
        with self.lock:
            if self.recording:
                self.audio_data.append(indata.copy())

    def start(self) -> None:
        """Open the microphone stream and begin collecting audio chunks."""
        self.audio_data = []
        self.recording  = True
        self.stream     = sd.InputStream(samplerate=self.config.samplerate, channels=1,
                                         dtype="int16", callback=self._audio_callback)
        self.stream.start()
        print("Recording... (press the shortcut again to stop)")

    def stop(self) -> npt.NDArray[np.int16]:
        """Stop recording, close the stream, and return all captured audio.

        Sleeps 100 ms after clearing the recording flag to allow any in-flight
        callback invocation to complete before the stream is closed.

        Returns:
            Concatenated audio as a 1-D int16 NumPy array, or an empty array
            (size == 0) if no audio was captured.
        """
        self.recording = False
        time.sleep(0.1)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.audio_data:
            return np.array([], dtype=np.int16)
        return np.concatenate(self.audio_data, axis=0)


class Transcriber:
    """Converts raw audio to text using the OpenAI Whisper API.

    Handles the temporary WAV file lifecycle: write, upload, delete.
    Exceptions from the API are propagated to the caller.
    """

    def __init__(self, client: OpenAI, config: Config) -> None:
        """Initialise the transcriber.

        Args:
            client: Authenticated OpenAI client instance.
            config: Application configuration supplying model, samplerate, and language.
        """
        self.client = client
        self.config = config

    def transcribe(self, audio: npt.NDArray[np.int16]) -> str:
        """Write audio to a temporary WAV file and send it to the Whisper API.

        The temporary file is always deleted in a finally block, even if the
        API call raises. Whitespace is stripped from the returned transcript.

        Args:
            audio: Raw PCM samples as a 1-D int16 NumPy array.

        Returns:
            The transcribed text with leading/trailing whitespace removed.

        Raises:
            Exception: Any error raised by the OpenAI API or file I/O.
        """
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            wav.write(tmp_path, self.config.samplerate, audio)

            print("Transcribing...")
            with open(tmp_path, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    model=self.config.model,
                    file=audio_file,
                    language=self.config.language,
                )
            text: str = result.text.strip()
            print(f"Transcribed: {text}")
            return text
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TextOutput:
    """Delivers transcribed text to the currently active window.

    Writes to the system clipboard and simulates Ctrl+V so the text appears
    wherever the cursor is focused, regardless of the application.
    """

    def paste(self, text: str) -> None:
        """Copy text to the clipboard and paste it into the active window.

        Args:
            text: The string to paste.
        """
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")


class HotkeyController:
    """Orchestrates the recording pipeline in response to hotkey presses.

    Owns the toggle logic and the daemon thread that runs the transcription
    pipeline asynchronously, so the hotkey listener is never blocked.
    """

    def __init__(
        self,
        audio_recorder: AudioRecorder,
        transcriber: Transcriber,
        text_output: TextOutput,
    ) -> None:
        """Wire together the three pipeline components.

        Args:
            audio_recorder: Captures microphone audio.
            transcriber: Converts audio to text via the Whisper API.
            text_output: Pastes the transcribed text into the active window.
        """
        self.audio_recorder = audio_recorder
        self.transcriber    = transcriber
        self.text_output    = text_output

    def _run_transcription_pipeline(self) -> None:
        """Stop recording, transcribe, and paste; run as a daemon thread target.

        Prints a warning if no audio was captured. Any exception raised by
        Transcriber or TextOutput is caught and printed so the thread exits
        cleanly without crashing the application.
        """
        try:
            audio = self.audio_recorder.stop()
            if audio.size == 0:
                print("No audio captured.")
                return
            text = self.transcriber.transcribe(audio)
            self.text_output.paste(text)
        except Exception as e:
            print(f"Error: {e}")

    def toggle(self, _event: object = None) -> None:
        """Start or stop recording depending on the current state.

        Called on every hotkey press. Reads the recording flag under the lock
        to avoid a race with the audio callback, then either starts recording
        directly or spawns a daemon thread to run the transcription pipeline.

        Args:
            _event: Keyboard event passed by the keyboard library (unused).
        """
        with self.audio_recorder.lock:
            is_recording = self.audio_recorder.recording
        if not is_recording:
            self.audio_recorder.start()
        else:
            threading.Thread(target=self._run_transcription_pipeline, daemon=True).start()

    def register(self, config: Config) -> None:
        """Bind the configured hotkey to toggle().

        Args:
            config: Application configuration supplying the hotkey string.
        """
        keyboard.add_hotkey(config.hotkey, self.toggle)


def main() -> None:
    """Entry point: compose the pipeline and block until Ctrl+C is pressed."""
    print("Whisper Voice Input active")
    print(f"   Shortcut : {config.hotkey.upper()}")
    print(f"   Language : {config.language}")
    print("   Press Ctrl+C to quit\n")

    client         = OpenAI(api_key=OPENAI_API_KEY)
    audio_recorder = AudioRecorder(config=config)
    transcriber    = Transcriber(client=client, config=config)
    text_output    = TextOutput()
    controller     = HotkeyController(audio_recorder, transcriber, text_output)
    controller.register(config)
    keyboard.wait("ctrl+c")


if __name__ == "__main__":
    main()
