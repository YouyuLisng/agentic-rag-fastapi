"""Inserts a batch of synthetic tours so the search_tours many-match path
(LIMIT + total-count reporting) can be exercised at a realistic scale --
the 8 curated seed tours never make a filter match more than a handful.

These rows are DEMO/stress-test data, not product data: every title is
prefixed with the BULK_MARKER so they're obvious in the UI and removable
in one command (`--clear`), and they're added on top of whatever seed.py
already loaded rather than replacing it. Deterministic (no randomness),
so a rerun reproduces the same catalog.

Generate a realistic travel catalog, stored only in this synthetic form
because the point is volume, not hand-authored content:
    uv run python -m app.scripts.seed_bulk_tours --count 90
Remove just these rows (leaves the 8 curated tours intact):
    uv run python -m app.scripts.seed_bulk_tours --clear
"""

import argparse
import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from app.db import close_pool, get_pool, init_pool  # noqa: E402

BULK_MARKER = "[壓測]"

# Must stay within the enums the tool schema advertises (app/agent/tools.py),
# or the generated rows would never match a well-formed filter.
COUNTRIES = ["日本", "泰國", "紐西蘭", "韓國", "印尼", "台灣"]
SUITABLE = [
    "一般大眾", "賞楓愛好者", "親子家庭", "長輩", "銀髮族", "學生", "小資族",
    "自然愛好者", "健行族", "年輕族群", "情侶", "蜜月", "樂齡族",
]
LOCATIONS = {
    "日本": ["東京", "大阪", "京都", "北海道", "沖繩"],
    "泰國": ["曼谷", "清邁", "普吉島", "蘇美島"],
    "紐西蘭": ["奧克蘭", "皇后鎮", "基督城"],
    "韓國": ["首爾", "釜山", "濟州島"],
    "印尼": ["峇里島", "雅加達", "日惹"],
    "台灣": ["花蓮", "台東", "墾丁", "阿里山"],
}


def _generate(count: int) -> list[dict]:
    tours: list[dict] = []
    for i in range(count):
        country = COUNTRIES[i % len(COUNTRIES)]
        locs = LOCATIONS[country]
        location = locs[i % len(locs)]
        days = 3 + (i % 8)  # 3..10
        budget = 18000 + (i % 15) * 4000  # 18000..74000
        suitable = SUITABLE[i % len(SUITABLE)]
        tours.append(
            {
                "title": f"{BULK_MARKER}{country}{location}{days}日遊 #{i + 1}",
                "country": country,
                "location": location,
                "days": days,
                "budget_twd": budget,
                "suitable_for": [suitable],
                "summary": f"{location}{days}日行程,適合{suitable},合成壓測資料。",
                "itinerary": [],
                "capacity": 20,
                "enrolled_count": i % 20,
                "cost_price_twd": int(budget * 0.7),
            }
        )
    return tours


async def _clear(conn) -> None:
    async with conn.cursor() as cur:
        await cur.execute("delete from tours where title like %s", (f"{BULK_MARKER}%",))
        deleted = cur.rowcount
    await conn.commit()
    print(f"Removed {deleted} bulk tours (curated tours untouched).")


async def _insert(conn, count: int) -> None:
    tours = _generate(count)
    async with conn.cursor() as cur:
        # Clear any prior bulk batch first so reruns don't pile up.
        await cur.execute("delete from tours where title like %s", (f"{BULK_MARKER}%",))
        for t in tours:
            await cur.execute(
                """
                insert into tours
                    (title, country, location, days, budget_twd, suitable_for, summary,
                     itinerary, capacity, enrolled_count, cost_price_twd)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    t["title"], t["country"], t["location"], t["days"], t["budget_twd"],
                    t["suitable_for"], t["summary"], json.dumps(t["itinerary"]),
                    t["capacity"], t["enrolled_count"], t["cost_price_twd"],
                ),
            )
    await conn.commit()
    per_country = count // len(COUNTRIES)
    print(f"Inserted {count} bulk tours (~{per_country}/country) marked {BULK_MARKER}.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed or clear synthetic bulk tours.")
    parser.add_argument("--count", type=int, default=90, help="How many bulk tours to insert.")
    parser.add_argument("--clear", action="store_true", help="Remove bulk tours instead of inserting.")
    args = parser.parse_args()

    await init_pool()
    pool = get_pool()
    try:
        async with pool.connection() as conn:
            if args.clear:
                await _clear(conn)
            else:
                await _insert(conn, args.count)
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
