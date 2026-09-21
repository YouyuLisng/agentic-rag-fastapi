"""Eval set for the third scoring layer JD reviewers actually look for:
did the agent pick the *right* tool(s), not just produce a plausible-
sounding answer. Faithfulness (generation eval) catches hallucinated
content; this catches routing mistakes -- calling search_tours for a
policy question, or not calling anything when a question genuinely
needs a live lookup.

expected_tools is a set: {} means no tool call is correct (small talk,
meta questions), and multiple entries mean a compound question that
should trigger more than one tool. Covers normal, compound, no-tool,
and error/hallucination-guard cases -- the last group specifically
targets "the agent must search before answering, and must not invent
a plausible-sounding result when the search comes back empty."
"""

ToolSelectionCase = dict[str, object]

TOOL_SELECTION_CASES: list[ToolSelectionCase] = [
    # -- 單一意圖:政策類 (search_knowledge) --
    {"query": "退訂政策是什麼?", "expected_tools": {"search_knowledge"}},
    {"query": "去日本需要辦簽證嗎?", "expected_tools": {"search_knowledge"}},
    {"query": "在國外受傷看醫生,保險有理賠嗎?", "expected_tools": {"search_knowledge"}},
    {"query": "報名後多久要付訂金?", "expected_tools": {"search_knowledge"}},
    {"query": "我吃素,可以特別安排餐食嗎?", "expected_tools": {"search_knowledge"}},
    # -- 單一意圖:行程篩選 (search_tours) --
    {"query": "有沒有適合長輩的行程?", "expected_tools": {"search_tours"}},
    {"query": "預算兩萬以內、五天左右的行程有哪些?", "expected_tools": {"search_tours"}},
    {"query": "日本有哪些團可以選?", "expected_tools": {"search_tours"}},
    {"query": "有沒有適合情侶的蜜月行程?", "expected_tools": {"search_tours"}},
    # -- 需要先查列表、再查細節/名額 (複合單輪) --
    {
        "query": "幫我查一下峇里島蜜月五日遊的完整每日行程",
        "expected_tools": {"search_tours", "get_tour_detail"},
    },
    {
        "query": "峇里島蜜月五日遊這團還有名額嗎?",
        "expected_tools": {"search_tours", "check_availability"},
    },
    # -- 跨兩種能力的複合問題 --
    {
        "query": "幫我找峇里島的蜜月行程,順便告訴我退訂政策是什麼",
        "expected_tools": {"search_tours", "search_knowledge"},
    },
    {
        "query": "北海道溫泉團要準備什麼衣物?這團預算多少?",
        "expected_tools": {"search_knowledge", "search_tours"},
    },
    # -- 不需要工具(寒暄/後設問題) --
    {"query": "你好,你是誰?", "expected_tools": set()},
    {"query": "你可以幫我做什麼?", "expected_tools": set()},
    {"query": "謝謝你的幫忙", "expected_tools": set()},
    # -- 邊界/錯誤案例(幻覺防範) --
    # 目的地不在服務範圍內,但仍應該先查詢確認,而不是直接憑印象回答
    # "沒有"或反過來瞎掰一個行程 -- 這是 JD 回饋裡點名的那種案例。
    {"query": "有沒有非洲的行程?", "expected_tools": {"search_tours"}},
    {"query": "有沒有預算三千元以下的行程?", "expected_tools": {"search_tours"}},
    {"query": "有沒有美國的行程?", "expected_tools": {"search_tours"}},
]
