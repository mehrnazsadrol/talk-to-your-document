from __future__ import annotations

from faster_whisper import WhisperModel

from app.config import settings

_model: WhisperModel | None = None


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        )
    return _model


def transcribe(path: str) -> tuple[list[dict], float, str]:
    """Returns (segments, duration_seconds, language).

    Language is an empty string if Whisper's detection failed.
    """
    model = _get_model()
    segments, info = model.transcribe(path)
    # materialise generator so info fields populate
    out = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
    language = info.language or ""
    return out, float(info.duration), language
