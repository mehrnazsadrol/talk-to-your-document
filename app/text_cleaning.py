"""PDF text cleaning applied before chunking.

Heuristic only. `strip_front_matter` drops a statute's table of contents and
legislative history by cutting everything before the first enacting-text
marker. The markers are legal-document specific and matched case-sensitively;
if none is found the text is returned unchanged, so non-statute PDFs are
unaffected. A document that legitimately contains one of these markers near
the top could be over-trimmed -- this is a tuned heuristic, not a general
front-matter detector.
"""

from __future__ import annotations

_FRONT_MATTER_ANCHORS = ("WHEREAS", "HER MAJESTY", "INTERPRETATION AND APPLICATION")


def strip_front_matter(text: str) -> str:
    positions = [p for p in (text.find(a) for a in _FRONT_MATTER_ANCHORS) if p != -1]
    if not positions:
        return text
    return text[min(positions) :]
