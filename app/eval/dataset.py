"""Ground-truth eval set for search_knowledge's retrieval quality:
each query is a paraphrase, not a copy of the source document's own
wording, since the whole point is testing whether embedding similarity
generalizes past exact keyword overlap. Two queries per policy
document for reasonable per-document signal without the set becoming
unwieldy to maintain by hand.
"""

EVAL_CASES: list[dict[str, str]] = [
    {"query": "如果我出發前 10 天想取消行程,可以退多少錢?", "expected_slug": "cancellation"},
    {"query": "退款大概要多久才會拿到?", "expected_slug": "cancellation"},
    {"query": "如果我在國外受傷需要看醫生,保險有理賠嗎?", "expected_slug": "insurance"},
    {"query": "行李在機場丟了怎麼辦?", "expected_slug": "insurance"},
    {"query": "報名後多久要付訂金?", "expected_slug": "payment"},
    {"query": "可以用信用卡分期付款嗎?", "expected_slug": "payment"},
    {"query": "護照效期有什麼要求?", "expected_slug": "preparation"},
    {"query": "出發前會有行前說明會嗎?", "expected_slug": "preparation"},
    {"query": "我有素食需求,可以特別安排餐食嗎?", "expected_slug": "special-needs"},
    {"query": "帶嬰兒出遊有推車可以租嗎?", "expected_slug": "special-needs"},
    {"query": "如果出發前遇到颱風,行程會怎麼處理?", "expected_slug": "force-majeure"},
    {"query": "因為疫情取消行程,費用會退嗎?", "expected_slug": "force-majeure"},
    {"query": "去日本玩需要辦簽證嗎?", "expected_slug": "visa"},
    {"query": "去紐西蘭要準備什麼證件?", "expected_slug": "visa"},
    {"query": "去北海道旅遊要準備什麼衣物?", "expected_slug": "packing"},
    {"query": "去熱帶國家玩要注意防曬嗎?", "expected_slug": "packing"},
]
