"""Eval set for generation-quality scoring (Faithfulness / Answer
Relevancy). One query per policy document, drawn from the same
hand-labeled set used for retrieval eval -- kept to 8 rather than the
full 16 because each case here runs a full agent turn plus two judge
calls, not just one embedding call.
"""

GENERATION_EVAL_QUERIES: list[str] = [
    "如果我出發前 10 天想取消行程,可以退多少錢?",
    "如果我在國外受傷需要看醫生,保險有理賠嗎?",
    "報名後多久要付訂金?",
    "護照效期有什麼要求?",
    "我有素食需求,可以特別安排餐食嗎?",
    "如果出發前遇到颱風,行程會怎麼處理?",
    "去日本玩需要辦簽證嗎?",
    "去北海道旅遊要準備什麼衣物?",
]
