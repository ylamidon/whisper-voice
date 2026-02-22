"""
Whisper Voice Input - Windows
==============================
Raccourci : Ctrl+Alt+Espace  →  démarre l'enregistrement
            Ctrl+Alt+Espace  →  arrête, transcrit, et colle le texte

Installation :
    pip install openai sounddevice scipy pyperclip keyboard pyautogui

Configuration :
    Mettre ta clé API dans la variable OPENAI_API_KEY ci-dessous
    ou dans la variable d'environnement OPENAI_API_KEY
"""

import os
import time
import tempfile
import threading

import sounddevice as sd
import scipy.io.wavfile as wav
import pyperclip
import keyboard
import pyautogui
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ─── CONFIG ───────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-...")   # ta clé ici
HOTKEY         = "ctrl+alt+space"
LANGUAGE       = "fr"          # français — change en "en" si besoin
SAMPLERATE     = 16000
MODEL          = "whisper-1"

# ──────────────────────────────────────────────────────────────────────────────

client    = OpenAI(api_key=OPENAI_API_KEY)
recording = False
audio_data = []
stream     = None
lock       = threading.Lock()


def audio_callback(indata, frames, time_info, status):
    with lock:
        if recording:
            audio_data.append(indata.copy())


def start_recording():
    global recording, audio_data, stream
    audio_data = []
    recording  = True
    stream     = sd.InputStream(samplerate=SAMPLERATE, channels=1,
                                 dtype="int16", callback=audio_callback)
    stream.start()
    print("🎙️  Enregistrement... (appuie à nouveau sur le raccourci pour arrêter)")


def stop_and_transcribe():
    global recording, stream

    recording = False
    time.sleep(0.1)
    if stream:
        stream.stop()
        stream.close()
        stream = None

    if not audio_data:
        print("⚠️  Aucun audio capturé.")
        return

    import numpy as np
    audio_np = np.concatenate(audio_data, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        wav.write(tmp_path, SAMPLERATE, audio_np)

    print("⏳ Transcription en cours...")
    try:
        with open(tmp_path, "rb") as audio_file:
            result = client.audio.transcriptions.create(
                model=MODEL,
                file=audio_file,
                language=LANGUAGE,
            )
        text = result.text.strip()
        print(f"✅ Transcrit : {text}")

        # Colle dans la fenêtre active
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")

    except Exception as e:
        print(f"❌ Erreur : {e}")
    finally:
        os.unlink(tmp_path)


def toggle(event=None):
    if not recording:
        start_recording()
    else:
        threading.Thread(target=stop_and_transcribe, daemon=True).start()


def main():
    print(f"✅ Whisper Voice Input actif")
    print(f"   Raccourci : {HOTKEY.upper()}")
    print(f"   Langue    : {LANGUAGE}")
    print(f"   Appuie sur Ctrl+C pour quitter\n")

    keyboard.add_hotkey(HOTKEY, toggle)
    keyboard.wait("ctrl+c")


if __name__ == "__main__":
    main()
