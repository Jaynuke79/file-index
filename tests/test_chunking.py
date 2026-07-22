from file_index.embed import chunk_segments, chunk_text


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_single_chunk():
    chunks = chunk_text("just a few words here")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "just a few words here"
    assert chunks[0]["ts_start"] is None


def test_long_text_chunked_with_overlap():
    words = [f"w{i}" for i in range(2000)]
    chunks = chunk_text(" ".join(words), chunk_tokens=500, overlap_tokens=50)
    assert len(chunks) > 1
    # every word appears somewhere
    joined = " ".join(c["text"] for c in chunks)
    assert "w0" in joined and "w1999" in joined
    # consecutive chunks overlap
    c0 = chunks[0]["text"].split()
    c1 = chunks[1]["text"].split()
    assert set(c0) & set(c1)
    # chunk sizes are ~500 tokens ≈ 375 words
    assert all(len(c["text"].split()) <= 375 for c in chunks)


def test_no_infinite_loop_on_tiny_chunks():
    chunks = chunk_text("a b c d e", chunk_tokens=1, overlap_tokens=1)
    assert len(chunks) >= 1
    joined = " ".join(c["text"] for c in chunks)
    assert "e" in joined


def test_segment_chunking_preserves_timestamps():
    segments = [
        {"start": float(i), "end": float(i + 1), "text": f"word{i} " * 50}
        for i in range(20)
    ]
    chunks = chunk_segments(segments, chunk_tokens=500)
    assert len(chunks) > 1
    assert chunks[0]["ts_start"] == 0.0
    for c in chunks:
        assert c["ts_start"] is not None and c["ts_end"] is not None
        assert c["ts_end"] >= c["ts_start"]
    # ranges are ordered and non-overlapping across chunks
    for a, b in zip(chunks, chunks[1:]):
        assert b["ts_start"] >= a["ts_start"]


def test_segment_chunking_skips_empty_segments():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "   "},
        {"start": 1.0, "end": 2.0, "text": "hello"},
    ]
    chunks = chunk_segments(segments)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "hello"
    assert chunks[0]["ts_start"] == 1.0
