"""Shared helper for parsing structured JSON out of an LLM's raw text
response -- used anywhere a prompt asks a model to "output only JSON"
and something needs to defensively parse what actually comes back.
"""


def extract_json(text: str) -> str:
    """Extract the first balanced top-level JSON object from raw model
    output. Models asked to output *only* JSON sometimes wrap it in a
    ```json fence anyway, or -- surfaced by a real eval run -- append
    trailing prose after the closing brace when the input driving the
    response is long/repetitive/unusual enough to nudge them into
    explaining themselves. Brace-depth counting that's aware of quoted
    strings (so a literal brace inside a value doesn't miscount) finds
    the actual object regardless of what surrounds it."""
    start = text.find("{")
    if start == -1:
        return text.strip()

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:].strip()
