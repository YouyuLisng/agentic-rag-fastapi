import json
from types import SimpleNamespace

import pytest

from app.eval import judge as judge_module
from app.eval.judge import _cosine_similarity, _extract_json, score_answer_relevancy, score_faithfulness


class FakeMessages:
    def __init__(self, raw_text: str):
        self._raw_text = raw_text

    async def create(self, **kwargs):
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._raw_text)])


class FakeClient:
    def __init__(self, raw_text: str):
        self.messages = FakeMessages(raw_text)


def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_extract_json_strips_markdown_fence():
    fenced = '```json\n{"a": 1}\n```'
    assert json.loads(_extract_json(fenced)) == {"a": 1}


def test_extract_json_passes_through_bare_json():
    assert _extract_json('{"a": 1}') == '{"a": 1}'


def test_extract_json_strips_trailing_prose_after_closing_brace():
    trailing = '{"claims": [{"claim": "退還 70%", "supported": true}]}\n\n這是根據退訂政策第二條計算的結果。'
    assert json.loads(_extract_json(trailing)) == {"claims": [{"claim": "退還 70%", "supported": True}]}


def test_extract_json_ignores_braces_inside_quoted_strings():
    # A claim whose own text contains literal { } must not desync the
    # brace-depth counter that finds the real closing brace.
    tricky = '{"claims": [{"claim": "設定 {變數} 後生效", "supported": false}]}'
    assert json.loads(_extract_json(tricky)) == {"claims": [{"claim": "設定 {變數} 後生效", "supported": False}]}


async def test_score_faithfulness_all_claims_supported():
    raw = json.dumps({"claims": [{"claim": "A", "supported": True}, {"claim": "B", "supported": True}]})
    client = FakeClient(raw)

    score, claims = await score_faithfulness(client, "judge-model", "answer text", ["some context"])

    assert score == 1.0
    assert len(claims) == 2


async def test_score_faithfulness_partial_support():
    raw = json.dumps({"claims": [{"claim": "A", "supported": True}, {"claim": "B", "supported": False}]})
    client = FakeClient(raw)

    score, _ = await score_faithfulness(client, "judge-model", "answer text", ["some context"])

    assert score == 0.5


async def test_score_faithfulness_no_claims_is_vacuously_faithful():
    raw = json.dumps({"claims": []})
    client = FakeClient(raw)

    score, claims = await score_faithfulness(client, "judge-model", "你好!", [])

    assert score == 1.0
    assert claims == []


async def test_score_answer_relevancy_noncommittal_scores_zero_without_embedding(
    monkeypatch: pytest.MonkeyPatch,
):
    raw = json.dumps({"questions": [], "noncommittal": True})
    client = FakeClient(raw)

    async def fail_embed(text: str):
        raise AssertionError("should not embed a noncommittal answer")

    monkeypatch.setattr(judge_module, "embed_query", fail_embed)

    score, questions, noncommittal = await score_answer_relevancy(client, "judge-model", "原始問題", "答非所問")

    assert score == 0.0
    assert questions == []
    assert noncommittal is True


async def test_score_answer_relevancy_averages_cosine_similarity(monkeypatch: pytest.MonkeyPatch):
    raw = json.dumps({"questions": ["q1", "q2"], "noncommittal": False})
    client = FakeClient(raw)

    embeddings = {
        "原始問題": [1.0, 0.0],
        "q1": [1.0, 0.0],  # identical -> similarity 1.0
        "q2": [0.0, 1.0],  # orthogonal -> similarity 0.0
    }

    async def fake_embed(text: str):
        return embeddings[text]

    monkeypatch.setattr(judge_module, "embed_query", fake_embed)

    score, questions, noncommittal = await score_answer_relevancy(client, "judge-model", "原始問題", "answer")

    assert score == pytest.approx(0.5)
    assert questions == ["q1", "q2"]
    assert noncommittal is False
