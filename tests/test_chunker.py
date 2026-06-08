from app.chunker import Chunk, chunk_text


def test_empty_string_returns_empty_list():
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    text = "Hello, world."
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)
    assert chunks[0].chunk_index == 0


def test_long_text_chunks_respect_size_and_overlap():
    paragraph = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    text = "\n\n".join([paragraph] * 10)

    chunk_size = 50
    chunk_overlap = 10
    chunk_size_chars = chunk_size * 4
    chunk_overlap_chars = chunk_overlap * 4

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(chunks) > 1
    for c in chunks:
        assert c.end_char - c.start_char <= chunk_size_chars + chunk_overlap_chars
        assert c.text == text[c.start_char : c.end_char]

    for prev, nxt in zip(chunks, chunks[1:]):
        expected_overlap = min(chunk_overlap_chars, prev.end_char - prev.start_char)
        assert nxt.start_char == prev.end_char - expected_overlap


def test_deterministic_same_input_same_output():
    text = ("Lorem ipsum dolor sit amet. " * 200).strip()
    a = chunk_text(text, chunk_size=50, chunk_overlap=10)
    b = chunk_text(text, chunk_size=50, chunk_overlap=10)
    assert a == b


def test_chunk_indices_sequential_and_offsets_consistent():
    text = ("Sentence number one. Sentence number two. " * 100).strip()
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert isinstance(c, Chunk)
        assert c.chunk_index == i
        assert 0 <= c.start_char < c.end_char <= len(text)
        assert c.text == text[c.start_char : c.end_char]

    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)
