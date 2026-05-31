# Sample audio for Whisper smoke tests

Drop a short (~10–15 s) English speech WAV at `data/samples/hello.wav`. It's
checked in so the smoke test below is reproducible across machines.

## Producing `hello.wav`

### macOS (built-in `say` + `ffmpeg`)

```bash
say "Hello world. This is a smoke test for faster whisper running on CPU." -o /tmp/hello.aiff
ffmpeg -i /tmp/hello.aiff -ar 16000 -ac 1 data/samples/hello.wav
```

### Any platform (record yourself with `ffmpeg`)

```bash
ffmpeg -f avfoundation -i ":0" -t 12 -ar 16000 -ac 1 data/samples/hello.wav   # macOS mic
ffmpeg -f alsa -i default -t 12 -ar 16000 -ac 1 data/samples/hello.wav        # Linux mic
```

`ffmpeg` must be on your `PATH` for non-WAV inputs; install via `brew install
ffmpeg` (macOS) or your distro's package manager.

## Smoke test

With the venv active and `pip install -r requirements.txt` done:

```bash
python - <<'PY'
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("data/samples/hello.wav")
print(f"language={info.language}  duration={info.duration:.2f}s")
for seg in segments:
    print(f"[{seg.start:.2f}-{seg.end:.2f}] {seg.text}")
PY
```

Expected on first run: ~140 MB model download, then timestamped segments,
a non-zero `info.duration`, and `info.language == "en"` for the sample above.
