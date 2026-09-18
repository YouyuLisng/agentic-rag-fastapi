def chunk_markdown(text: str, max_chars: int = 500) -> list[str]:
    """Greedily group consecutive paragraphs up to max_chars.

    These policy docs are short, hand-written prose, not large PDFs --
    splitting mid-paragraph would cut a single idea in half for no
    benefit. A paragraph longer than max_chars on its own is kept whole
    rather than force-split, trading a bit of size consistency for
    semantic coherence.
    """
    paragraphs = [p.strip() for p in text.strip().split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if not current or len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks
