from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

from pydub import AudioSegment as PydubAudioSegment
from pydub.silence import detect_nonsilent

from app import transcriber
from app.config import settings


@dataclass
class AudioSegment:
    text: str
    start_seconds: float
    end_seconds: float
    segment_index: int


def _export_slice(audio: PydubAudioSegment, start_ms: int, end_ms: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    slice_audio = (
        audio[start_ms:end_ms].set_frame_rate(16000).set_channels(1).set_sample_width(2)
    )
    slice_audio.export(tmp.name, format="wav")
    return tmp.name


def split_audio(path: str) -> list[tuple[float, float, str]]:
    audio = PydubAudioSegment.from_file(path)
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=settings.audio_min_silence_ms,
        silence_thresh=settings.audio_silence_thresh_db,
    )

    if not nonsilent:
        return []

    total_ms = len(audio)
    max_ms = settings.audio_max_segment_seconds * 1000

    if len(nonsilent) == 1 and total_ms > max_ms:
        out: list[tuple[float, float, str]] = []
        for start_ms in range(0, total_ms, max_ms):
            end_ms = min(start_ms + max_ms, total_ms)
            tmp_path = _export_slice(audio, start_ms, end_ms)
            out.append((start_ms / 1000.0, end_ms / 1000.0, tmp_path))
        return out

    out = []
    for start_ms, end_ms in nonsilent:
        tmp_path = _export_slice(audio, start_ms, end_ms)
        out.append((start_ms / 1000.0, end_ms / 1000.0, tmp_path))
    return out


def transcribe_segments(path: str) -> tuple[list[AudioSegment], float, str]:
    duration_seconds = len(PydubAudioSegment.from_file(path)) / 1000.0
    slices = split_audio(path)

    results: list[AudioSegment] = []
    language = ""
    try:
        for i, (start_s, end_s, slice_path) in enumerate(slices):
            sub_segments, _slice_duration, slice_language = transcriber.transcribe(
                slice_path
            )
            if i == 0:
                language = slice_language
            text = " ".join(s["text"] for s in sub_segments).strip()
            results.append(
                AudioSegment(
                    text=text,
                    start_seconds=start_s,
                    end_seconds=end_s,
                    segment_index=i,
                )
            )
    finally:
        for _start, _end, slice_path in slices:
            try:
                os.unlink(slice_path)
            except OSError:
                pass

    return results, duration_seconds, language
