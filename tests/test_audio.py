import os

import pytest

from app import audio as audio_mod
from app import transcriber


class _FakePydubAudio:
    """Mimics just enough of pydub.AudioSegment for split_audio."""

    def __init__(self, length_ms: int):
        self._length_ms = length_ms

    def __len__(self):
        return self._length_ms

    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop if key.stop is not None else self._length_ms
            return _FakePydubAudio(stop - start)
        raise TypeError(key)

    def set_frame_rate(self, _rate):
        return self

    def set_channels(self, _channels):
        return self

    def set_sample_width(self, _width):
        return self

    def export(self, path, format):  # noqa: A002 - matches pydub signature
        # Touch the file so tempfile-cleanup assertions are meaningful.
        with open(path, "wb") as f:
            f.write(b"RIFF")


def _install_pydub_stub(monkeypatch, length_ms: int, nonsilent_ranges: list):
    fake_audio = _FakePydubAudio(length_ms)
    monkeypatch.setattr(
        audio_mod, "PydubAudioSegment",
        type("PS", (), {"from_file": staticmethod(lambda _path: fake_audio)}),
    )
    monkeypatch.setattr(
        audio_mod, "detect_nonsilent",
        lambda _audio, min_silence_len, silence_thresh: list(nonsilent_ranges),
    )


def _install_transcriber_stub(monkeypatch, texts_per_call=None, language="en"):
    """Stub transcriber.transcribe to return a fixed text per call."""
    calls = {"i": 0}
    if texts_per_call is None:
        texts_per_call = ["stub text"]

    def fake_transcribe(_path):
        i = calls["i"]
        calls["i"] += 1
        text = texts_per_call[i] if i < len(texts_per_call) else texts_per_call[-1]
        return [{"start": 0.0, "end": 1.0, "text": text}], 1.0, language

    monkeypatch.setattr(transcriber, "transcribe", fake_transcribe)
    return calls


def test_split_audio_maps_ms_to_seconds(monkeypatch):
    _install_pydub_stub(
        monkeypatch,
        length_ms=10_000,
        nonsilent_ranges=[(0, 1500), (2000, 3500), (4000, 6000)],
    )

    out = audio_mod.split_audio("ignored.wav")
    try:
        bounds = [(s, e) for s, e, _p in out]
        assert bounds == [(0.0, 1.5), (2.0, 3.5), (4.0, 6.0)]
        for _s, _e, p in out:
            assert os.path.exists(p)
    finally:
        for _s, _e, p in out:
            if os.path.exists(p):
                os.unlink(p)


def test_split_audio_returns_empty_when_no_voiced_ranges(monkeypatch):
    _install_pydub_stub(monkeypatch, length_ms=10_000, nonsilent_ranges=[])
    assert audio_mod.split_audio("ignored.wav") == []


def test_split_audio_fallback_fixed_window_for_single_giant_block(monkeypatch):
    # One nonsilent block covering most of a 75s file; max segment is 30s.
    _install_pydub_stub(
        monkeypatch,
        length_ms=75_000,
        nonsilent_ranges=[(0, 75_000)],
    )

    out = audio_mod.split_audio("ignored.wav")
    try:
        bounds = [(s, e) for s, e, _p in out]
        assert bounds == [(0.0, 30.0), (30.0, 60.0), (60.0, 75.0)]
    finally:
        for _s, _e, p in out:
            if os.path.exists(p):
                os.unlink(p)


def test_transcribe_segments_indices_text_and_monotonic(monkeypatch):
    _install_pydub_stub(
        monkeypatch,
        length_ms=10_000,
        nonsilent_ranges=[(0, 1500), (2000, 3500), (4000, 6000)],
    )
    _install_transcriber_stub(monkeypatch, texts_per_call=["one", "two", "three"], language="en")

    segments, duration, language = audio_mod.transcribe_segments("ignored.wav")

    assert language == "en"
    assert duration == 10.0
    assert [s.text for s in segments] == ["one", "two", "three"]
    assert [s.segment_index for s in segments] == [0, 1, 2]
    starts = [s.start_seconds for s in segments]
    assert starts == sorted(starts)
    assert [s.start_seconds for s in segments] == [0.0, 2.0, 4.0]
    assert [s.end_seconds for s in segments] == [1.5, 3.5, 6.0]


def test_transcribe_segments_empty_when_silent(monkeypatch):
    _install_pydub_stub(monkeypatch, length_ms=10_000, nonsilent_ranges=[])
    _install_transcriber_stub(monkeypatch)

    segments, duration, language = audio_mod.transcribe_segments("ignored.wav")
    assert segments == []
    assert duration == 10.0
    assert language == ""


def test_transcribe_segments_cleans_up_tempfiles_on_success(monkeypatch):
    _install_pydub_stub(
        monkeypatch,
        length_ms=10_000,
        nonsilent_ranges=[(0, 1500), (2000, 3500)],
    )

    created_paths: list[str] = []
    orig_export_slice = audio_mod._export_slice

    def tracking_export(audio, start_ms, end_ms):
        p = orig_export_slice(audio, start_ms, end_ms)
        created_paths.append(p)
        return p

    monkeypatch.setattr(audio_mod, "_export_slice", tracking_export)
    _install_transcriber_stub(monkeypatch)

    audio_mod.transcribe_segments("ignored.wav")

    assert created_paths, "expected at least one tempfile to be created"
    for p in created_paths:
        assert not os.path.exists(p), f"tempfile not cleaned up: {p}"


def test_transcribe_segments_cleans_up_tempfiles_on_raise(monkeypatch):
    _install_pydub_stub(
        monkeypatch,
        length_ms=10_000,
        nonsilent_ranges=[(0, 1500), (2000, 3500)],
    )

    created_paths: list[str] = []
    orig_export_slice = audio_mod._export_slice

    def tracking_export(audio, start_ms, end_ms):
        p = orig_export_slice(audio, start_ms, end_ms)
        created_paths.append(p)
        return p

    monkeypatch.setattr(audio_mod, "_export_slice", tracking_export)

    def boom(_path):
        raise RuntimeError("whisper exploded")

    monkeypatch.setattr(transcriber, "transcribe", boom)

    with pytest.raises(RuntimeError):
        audio_mod.transcribe_segments("ignored.wav")

    assert created_paths
    for p in created_paths:
        assert not os.path.exists(p), f"tempfile not cleaned up: {p}"
