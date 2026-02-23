# Code Quality Improvement TODO

## Testing

- [ ] **Add test coverage for `main()`** — The entry point function has no tests. Test that it wires components correctly and handles `KeyboardInterrupt` gracefully.
- [ ] **Move tests into a `tests/` directory** — `test_whisper_voice.py` sits at the root. A `tests/` package is the conventional layout and keeps the root clean.
- [ ] **Add integration/end-to-end test** — All tests are unit tests with mocks. A test that runs the real `AudioRecorder.start()` / `stop()` cycle (even briefly) would catch wiring issues.
- [ ] **Test the `_audio_callback` status flag** — The `_status` parameter from sounddevice is silently ignored. Consider logging when `_status` reports an overflow/underflow, and test that path.

## Robustness & Error Handling

- [ ] **Guard against double-start / double-stop** — Rapidly pressing the hotkey can call `start()` while already recording or `stop()` after already stopped. Add a state check or make `toggle()` re-entrant safe.
- [ ] **Handle `sounddevice` device errors** — If no microphone is available, `sd.InputStream()` raises. `start()` should catch this and log a user-friendly message instead of crashing silently inside the hotkey callback.
- [ ] **Restore the clipboard after paste** — `TextOutput.paste()` overwrites whatever the user had in the clipboard. Save and restore the previous clipboard content.
- [ ] **Add a small delay before `Ctrl+V`** — Some applications need a brief pause between clipboard write and paste. A configurable delay (e.g. 50ms) in `TextOutput.paste()` would improve reliability.

## Configuration & Flexibility

- [ ] **Load config from environment / CLI args** — `Config` is hardcoded. Allow overriding `LANGUAGE`, `HOTKEY`, etc. via env vars or command-line arguments (e.g. using `argparse` or reading from `.env`).
- [ ] **Add a `__main__.py` entry point** — Allow running via `python -m whisper_voice` by adding a `__main__.py`, and declare a `[project.scripts]` entry in `pyproject.toml`.

## Code Structure

- [ ] **Use a protocol/ABC for `TextOutput`** — Defining a `TextOutputProtocol` would make it easy to swap in alternative output strategies (e.g. typing simulation instead of clipboard paste) and simplify testing.
- [ ] **Extract `_status` logging in audio callback** — Instead of ignoring `_status: sd.CallbackFlags`, check `_status` and log warnings on audio overflow/underflow. This helps diagnose recording issues.

## Tooling & CI

- [ ] **Add `ruff format` to CI** — The CI runs `ruff check` but not `ruff format --check`. Adding it ensures consistent formatting.
- [ ] **Enforce a minimum coverage threshold** — CI runs `--cov` but doesn't fail on low coverage. Add `--cov-fail-under=80` (or similar) to `pyproject.toml` pytest config.
- [ ] **Add a `.gitignore`** — Ensure common Python artifacts (`.venv/`, `__pycache__/`, `*.pyc`, `.env`) are ignored.
- [ ] **Run mypy on tests too** — CI only runs `mypy whisper_voice.py`. Extending it to `conftest.py` and `test_whisper_voice.py` catches type issues in test code.

## UX

- [ ] **Add audio feedback (beep/sound) on toggle** — A short beep when recording starts/stops gives the user confidence the hotkey was registered, since there's no visual indicator.
- [ ] **Add a system tray icon** — A tray icon showing recording state would make the tool more user-friendly than a terminal-only experience.
