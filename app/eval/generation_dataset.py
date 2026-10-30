"""Eval set for generation-quality scoring (Faithfulness / Answer
Relevancy). Three queries per policy document (24 total): the first two
are the same phrasings used in the retrieval eval set, the third probes
a different fact within the same document -- this exercises more of
each document's content, not just whichever paragraph the first query
already happens to touch, without ballooning cost the way matching the
full tool-selection set (19+) would (each case here runs a full agent
turn plus two judge calls, not just one embedding call).
"""

GENERATION_EVAL_QUERIES: list[str] = [
    # -- cancellation --
    "如果我出發前 10 天想取消行程,可以退多少錢?",
    "退款大概要多久才會拿到?",
    "如果公司因為成團人數不足取消行程,我的錢會怎麼處理?",
    # -- insurance --
    "如果我在國外受傷需要看醫生,保險有理賠嗎?",
    "行李在機場丟了怎麼辦?",
    "保險理賠申請大概要準備哪些文件?",
    # -- payment --
    "報名後多久要付訂金?",
    "可以用信用卡分期付款嗎?",
    "公司報名可以單獨開立統一發票嗎?",
    # -- preparation --
    "護照效期有什麼要求?",
    "出發前會有行前說明會嗎?",
    "去韓國玩需要自備電壓轉接頭嗎?",
    # -- special-needs --
    "我有素食需求,可以特別安排餐食嗎?",
    "帶嬰兒出遊有推車可以租嗎?",
    "家裡長輩行動比較不方便,可以報名一般行程嗎?",
    # -- force-majeure --
    "如果出發前遇到颱風,行程會怎麼處理?",
    "因為疫情取消行程,費用會退嗎?",
    "因為自己臨時生病取消行程,算是不可抗力嗎?",
    # -- visa --
    "去日本玩需要辦簽證嗎?",
    "去紐西蘭要準備什麼證件?",
    "去韓國玩需要事先申請電子旅行許可嗎?",
    # -- packing --
    "去北海道旅遊要準備什麼衣物?",
    "去熱帶國家玩要注意防曬嗎?",
    "去紐西蘭健行要準備什麼鞋子?",
]
