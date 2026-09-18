from app.rag.chunking import chunk_markdown


def test_empty_input_returns_no_chunks():
    assert chunk_markdown("") == []
    assert chunk_markdown("   \n\n  ") == []


def test_short_text_stays_in_one_chunk():
    text = "第一段。\n\n第二段。"
    assert chunk_markdown(text, max_chars=500) == ["第一段。\n\n第二段。"]


def test_splits_once_max_chars_exceeded():
    para_a = "A" * 300
    para_b = "B" * 300
    chunks = chunk_markdown(f"{para_a}\n\n{para_b}", max_chars=500)
    assert chunks == [para_a, para_b]


def test_oversized_single_paragraph_is_kept_whole_not_force_split():
    huge = "X" * 900
    assert chunk_markdown(huge, max_chars=500) == [huge]


def test_groups_multiple_small_paragraphs_into_one_chunk_up_to_limit():
    paras = ["短句一。", "短句二。", "短句三。"]
    chunks = chunk_markdown("\n\n".join(paras), max_chars=500)
    assert len(chunks) == 1
    assert all(p in chunks[0] for p in paras)
