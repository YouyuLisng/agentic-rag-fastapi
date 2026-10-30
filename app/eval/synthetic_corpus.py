"""Deterministic generator for a synthetic policy corpus built to stress
two-stage retrieval, plus its eval set.

Why synthetic, and why this shape: the 8 real policy docs are topically
distinct, so vector search already ceilings on them (accuracy@5 = 100%)
and a reranker has no headroom to prove itself. Reranking earns its keep
when the candidate set contains *near-duplicate distractors* -- documents
that are semantically almost identical except for one discriminating
entity. This corpus manufactures exactly that: each "family" is one
policy template instantiated over ~20 entities (product types, scenarios,
countries), so every doc in a family reads nearly the same to an
embedding model, and only a cross-encoder reading query+document together
reliably separates them. That is the honest place to measure what rerank
buys -- not a toy set where the first stage never errs.

Fully deterministic (no LLM, no randomness): reproducible, free, and
unaffected by API credit. It is DEMO/benchmark data -- clearly labelled
and kept in its own table (synthetic_policy_chunks), never mixed into the
real knowledge base.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticDoc:
    slug: str
    cluster: str
    title: str
    content: str


@dataclass(frozen=True)
class SyntheticEvalCase:
    query: str
    expected_slug: str
    cluster: str


# Each family: (cluster key, human title, list of discriminating entities,
# a template for the doc body, a template for the matching query). The
# template text is near-identical across entities within a family -- that
# sameness is the point (it's what makes the first stage confuse them).
_FAMILIES: list[tuple[str, str, list[str], str, str]] = [
    (
        "refund",
        "退款規則",
        ["國內團", "國外團", "郵輪行程", "機票自由行", "包機團", "蜜月客製團",
         "學生團", "銀髮團", "商務考察團", "遊學團", "登山團", "潛水團",
         "賞櫻團", "滑雪團", "親子團", "攝影團", "朝聖團", "美食團",
         "鐵道團", "遊輪岸上團"],
        "本條說明「{e}」的退款規則。出發前 30 天以上取消,退還全額扣除手續費;"
        "出發前 15 至 29 天取消,退還七成;出發前 7 至 14 天取消,退還五成;"
        "出發前 7 天內取消,恕不退款。「{e}」的退款一律依本條辦理,不適用其他產品線的級距。",
        "請問「{e}」出發前 10 天取消的話,費用可以退多少?",
    ),
    (
        "insurance-claim",
        "保險理賠項目",
        ["班機延誤", "行李遺失", "行李延誤", "海外醫療", "意外身故", "旅程取消",
         "旅程縮短", "緊急醫療轉送", "信用卡盜刷", "租車自負額", "個人責任",
         "現金竊盜", "護照遺失", "食物中毒", "運動傷害", "居家竊盜",
         "寵物寄宿", "慰問金", "搜救費用", "法律費用"],
        "本條說明「{e}」的理賠方式。申請時須於事件發生後 30 日內檢附證明文件,"
        "經核實後依保單約定給付。「{e}」的理賠額度與自負額以本條為準,"
        "與其他理賠項目分開計算,不得合併或互相挪用。",
        "「{e}」這一項的理賠要怎麼申請、賠多少?",
    ),
    (
        "visa",
        "簽證規定",
        ["日本", "泰國", "紐西蘭", "韓國", "印尼", "越南", "美國", "申根區",
         "澳洲", "印度", "加拿大", "英國", "菲律賓", "馬來西亞", "土耳其",
         "埃及", "南非", "巴西", "阿聯酋", "中國大陸"],
        "本條說明前往「{e}」的簽證規定。持中華民國護照者,"
        "依「{e}」現行規定辦理入境簽證或免簽,停留天數與可否落地簽依「{e}」官方公告為準。"
        "本社僅就「{e}」提供文件協助,最終核發與否由「{e}」主管機關決定。",
        "去「{e}」玩,簽證要怎麼辦?可以免簽嗎?",
    ),
    (
        "deposit",
        "訂金與尾款",
        ["國內團", "國外團", "郵輪行程", "機票自由行", "包機團", "蜜月客製團",
         "學生團", "銀髮團", "商務考察團", "遊學團", "登山團", "潛水團",
         "賞櫻團", "滑雪團", "親子團", "攝影團", "朝聖團", "美食團",
         "鐵道團", "遊輪岸上團"],
        "本條說明「{e}」的訂金與尾款規定。報名「{e}」須先繳訂金以保留名額,"
        "尾款須於出發前一定期限前結清,逾期視同放棄名額。「{e}」的訂金成數與繳款期限"
        "依本條辦理,與退款級距分屬不同條款。",
        "報名「{e}」的訂金要繳多少、尾款什麼時候要付清?",
    ),
    (
        "date-change",
        "更改與轉讓",
        ["國內團", "國外團", "郵輪行程", "機票自由行", "包機團", "蜜月客製團",
         "學生團", "銀髮團", "商務考察團", "遊學團", "登山團", "潛水團",
         "賞櫻團", "滑雪團", "親子團", "攝影團", "朝聖團", "美食團",
         "鐵道團", "遊輪岸上團"],
        "本條說明「{e}」的日期更改與名額轉讓。「{e}」如需改期,"
        "須在期限內提出並補足差價;如需將名額轉讓他人,須經本社書面同意並更新證件資料。"
        "「{e}」的改期與轉讓一律依本條,不得援引退款或訂金條款主張。",
        "「{e}」報名後可以改日期或換人參加嗎?要注意什麼?",
    ),
    (
        "baggage",
        "行李規定",
        ["經濟艙國際線", "商務艙國際線", "頭等艙國際線", "國內線", "廉價航空",
         "包機航班", "區域螺旋槳機", "郵輪隨身", "郵輪托運", "高鐵接駁",
         "遊覽車接駁", "自駕租車", "鐵道周遊", "登山裝備", "潛水裝備",
         "滑雪裝備", "嬰兒推車", "輪椅", "運動器材", "樂器"],
        "本條說明「{e}」的行李規定。「{e}」的免費托運與隨身件數、重量與尺寸上限"
        "依本條辦理,超過部分須加購或另計費用。「{e}」的特殊物品須事先申報,"
        "額度不與其他艙等或交通方式共用。",
        "「{e}」可以帶多少行李?超重或特殊物品怎麼算?",
    ),
    (
        "payment-method",
        "付款方式",
        ["信用卡一次", "信用卡分期", "銀行轉帳", "ATM 匯款", "超商代碼", "行動支付",
         "旅遊金折抵", "禮券折抵", "公司請款", "現金臨櫃", "外幣帳戶", "電子錢包",
         "分期零利率", "紅利折抵", "聯名卡優惠", "海外刷卡", "預付卡", "員工福利金",
         "工會補助", "政府補助"],
        "本條說明使用「{e}」付款的規定。以「{e}」支付團費時,"
        "到帳確認時間、可否取消或退刷、以及是否收取手續費,依本條辦理。"
        "「{e}」的相關規定與其他付款方式分列,不互相適用。",
        "我想用「{e}」付團費,有什麼限制或手續費嗎?",
    ),
    (
        "special-needs",
        "特殊需求協助",
        ["素食餐", "清真餐", "無麩質餐", "低鈉餐", "嬰兒餐", "輪椅協助",
         "聽障協助", "視障協助", "行動不便協助", "孕婦協助", "慢性病用藥",
         "過敏原告知", "宗教祈禱室", "寵物同行", "導盲犬同行", "氧氣機需求",
         "擔架需求", "隨行看護", "無障礙房", "兒童座椅"],
        "本條說明「{e}」的安排方式。旅客如需「{e}」,"
        "須於報名時或出發前一定期限提出,由本社代為向航空公司、飯店或餐廳確認。"
        "「{e}」能否滿足視各供應商量能而定,本條僅規範「{e}」的申請與告知流程。",
        "我需要安排「{e}」,請問要怎麼申請、來得及嗎?",
    ),
]


def _slug(cluster: str, index: int) -> str:
    return f"{cluster}-{index:02d}"


def generate_corpus() -> list[SyntheticDoc]:
    docs: list[SyntheticDoc] = []
    for cluster, title, entities, body_tmpl, _q in _FAMILIES:
        for i, entity in enumerate(entities):
            docs.append(
                SyntheticDoc(
                    slug=_slug(cluster, i),
                    cluster=cluster,
                    title=f"{title} -- {entity}",
                    content=body_tmpl.format(e=entity),
                )
            )
    return docs


# Eval cases: for each family, probe a spread of variants (every 4th, so
# ~5 per family) -- enough signal to move accuracy@1/MRR without firing a
# rerank call for all ~160 docs. Each query's correct answer is one
# specific variant sitting among ~19 near-identical siblings.
_EVAL_STRIDE = 4


def generate_eval_cases() -> list[SyntheticEvalCase]:
    cases: list[SyntheticEvalCase] = []
    for cluster, _title, entities, _body, query_tmpl in _FAMILIES:
        for i in range(0, len(entities), _EVAL_STRIDE):
            cases.append(
                SyntheticEvalCase(
                    query=query_tmpl.format(e=entities[i]),
                    expected_slug=_slug(cluster, i),
                    cluster=cluster,
                )
            )
    return cases
