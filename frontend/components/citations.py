from __future__ import annotations

import streamlit as st


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    if total >= 3600:
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"


def format_locator(metadata: dict) -> str:
    source_type = metadata.get("source_type")
    if source_type == "audio":
        start = metadata.get("start_seconds", 0)
        end = metadata.get("end_seconds", 0)
        return f"{format_timestamp(start)}–{format_timestamp(end)}"
    if source_type == "pdf":
        page = metadata.get("page_number")
        if page is None:
            return ""
        return f"p. {page}"
    return ""


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for src in sources:
            locator = format_locator(src)
            heading = f"**{src['source_filename']}**"
            if locator:
                heading = f"{heading} · {locator}"
            st.markdown(heading)
            st.caption(src["snippet"])
            st.caption(f"distance: {src['distance']:.3f}")
