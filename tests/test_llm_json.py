import json

from app.llm_json import extract_json


def test_extract_json_strips_markdown_fence():
    fenced = '```json\n{"a": 1}\n```'
    assert json.loads(extract_json(fenced)) == {"a": 1}


def test_extract_json_passes_through_bare_json():
    assert extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_strips_trailing_prose_after_closing_brace():
    trailing = '{"claims": [{"claim": "退還 70%", "supported": true}]}\n\n這是根據退訂政策第二條計算的結果。'
    assert json.loads(extract_json(trailing)) == {"claims": [{"claim": "退還 70%", "supported": True}]}


def test_extract_json_ignores_braces_inside_quoted_strings():
    # A value whose own text contains literal { } must not desync the
    # brace-depth counter that finds the real closing brace.
    tricky = '{"claims": [{"claim": "設定 {變數} 後生效", "supported": false}]}'
    assert json.loads(extract_json(tricky)) == {"claims": [{"claim": "設定 {變數} 後生效", "supported": False}]}


def test_extract_json_no_opening_brace_returns_stripped_text():
    assert extract_json("  not json at all  ") == "not json at all"
