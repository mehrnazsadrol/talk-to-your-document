"""Recursive character chunker.

Splits raw text into approximately `chunk_size`-token chunks with
approximately `chunk_overlap`-token overlap, using a deterministic
recursive split on a fixed separator hierarchy.

Token approximation: ~4 chars per token. `chunk_size` and `chunk_overlap`
are interpreted as token counts by the caller and multiplied by 4
internally to get character budgets.

`start_char`/`end_char` are offsets into the original input text; the
overlap window is realized by setting chunk N+1's `start_char` to
`chunk_N.end_char - overlap_chars` (not by string concatenation), so
`text == original[start_char:end_char]` always holds.
"""

from __future__ import annotations

from dataclasses import dataclass

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    chunk_index: int


def _split_offsets(
    text: str, start: int, end: int, sep_idx: int, max_chars: int
) -> list[tuple[int, int]]:
    if end - start <= max_chars:
        return [(start, end)]

    if sep_idx >= len(_SEPARATORS):
        return [(i, min(i + max_chars, end)) for i in range(start, end, max_chars)]

    sep = _SEPARATORS[sep_idx]

    if sep == "":
        return [(i, min(i + max_chars, end)) for i in range(start, end, max_chars)]

    region = text[start:end]
    pieces: list[tuple[int, int]] = []
    cursor = start
    pos = 0
    while True:
        idx = region.find(sep, pos)
        if idx == -1:
            pieces.append((cursor, end))
            break
        piece_end = start + idx + len(sep)
        pieces.append((cursor, piece_end))
        cursor = piece_end
        pos = idx + len(sep)

    if len(pieces) == 1:
        return _split_offsets(text, start, end, sep_idx + 1, max_chars)

    out: list[tuple[int, int]] = []
    for p_start, p_end in pieces:
        if p_end - p_start <= max_chars:
            out.append((p_start, p_end))
        else:
            out.extend(_split_offsets(text, p_start, p_end, sep_idx + 1, max_chars))
    return out


def chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[Chunk]:
    """Split `text` into chunks of approximately `chunk_size` tokens with
    approximately `chunk_overlap` tokens of overlap between adjacent chunks.

    Tokens are approximated as `chars / 4`; internally we work with
    `chunk_size * 4` and `chunk_overlap * 4` character budgets.

    Pure function — no I/O. Deterministic: same input always yields the
    same output.
    """
    if not text:
        return []

    chunk_size_chars = chunk_size * 4
    chunk_overlap_chars = chunk_overlap * 4

    if len(text) <= chunk_size_chars:
        return [Chunk(text=text, start_char=0, end_char=len(text), chunk_index=0)]

    spans = _split_offsets(text, 0, len(text), 0, chunk_size_chars)

    merged: list[tuple[int, int]] = []
    cur_start, cur_end = spans[0]
    for s, e in spans[1:]:
        if e - cur_start <= chunk_size_chars:
            cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    chunks: list[Chunk] = []
    for i, (s, e) in enumerate(merged):
        if i == 0:
            start = s
        else:
            prev_end = merged[i - 1][1]
            start = max(merged[i - 1][0], prev_end - chunk_overlap_chars)
        chunks.append(
            Chunk(
                text=text[start:e],
                start_char=start,
                end_char=e,
                chunk_index=i,
            )
        )
    return chunks
