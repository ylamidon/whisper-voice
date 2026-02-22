"""
Whisper Voice Input - Windows
==============================
Shortcut : Ctrl+Alt+Space  →  starts recording
           Ctrl+Alt+Space  →  stops, transcribes, and pastes the text

Installation:
    uv sync

Configuration:
    Put your API key in a .env file: OPENAI_API_KEY=sk-...
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
    hotkey:     str = "ctrl+alt+space"
    language:   str = "fr"          # French — change to "en" if needed
    samplerate: int = 16000
    model:      str = "whisper-1"

config = Config()

# ──────────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    sys.exit("OPENAI_API_KEY missing. Create a .env file with OPENAI_API_KEY=sk-...")


class Recorder:
    def __init__(self, client: OpenAI) -> None:
        self.client    = client
        self.recording: bool = False
        self.audio_data: list[npt.NDArray[np.int16]] = []
        self.stream:    Optional[sd.InputStream] = None
        self.lock       = threading.Lock()

    def audio_callback(
        self,
        indata: npt.NDArray[np.int16],
        _frames: int,
        _time_info: object,
        _status: sd.CallbackFlags,
    ) -> None:
        with self.lock:
            if self.recording:
                self.audio_data.append(indata.copy())

    def start_recording(self) -> None:
        self.audio_data = []
        self.recording  = True
        self.stream     = sd.InputStream(samplerate=config.samplerate, channels=1,
                                         dtype="int16", callback=self.audio_callback)
        self.stream.start()
        print("Recording... (press the shortcut again to stop)")

    def stop_and_transcribe(self) -> None:
        self.recording = False
        time.sleep(0.1)
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if not self.audio_data:
            print("No audio captured.")
            return

        audio_np: npt.NDArray[np.int16] = np.concatenate(self.audio_data, axis=0)

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            wav.write(tmp_path, config.samplerate, audio_np)

            print("Transcribing...")
            with open(tmp_path, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    model=config.model,
                    file=audio_file,
                    language=config.language,
                )
            text: str = result.text.strip()
            print(f"Transcribed: {text}")

            # Paste into the active window
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def toggle(self, _event: object = None) -> None:
        with self.lock:
            is_recording = self.recording
        if not is_recording:
            self.start_recording()
        else:
            threading.Thread(target=self.stop_and_transcribe, daemon=True).start()


def main() -> None:
    print("Whisper Voice Input active")
    print(f"   Shortcut : {config.hotkey.upper()}")
    print(f"   Language : {config.language}")
    print("   Press Ctrl+C to quit\n")

    recorder = Recorder(client=OpenAI(api_key=OPENAI_API_KEY))
    keyboard.add_hotkey(config.hotkey, recorder.toggle)
    keyboard.wait("ctrl+c")


if __name__ == "__main__":
    main()
