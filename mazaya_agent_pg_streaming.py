#!/usr/bin/env python3
"""
Mazaya Benefits Agent (LangGraph + LangMem + Postgres) — ASYNC, STREAMING

CHANGES IN THIS REV:
- Fixed Postgres pool lifecycle: explicit open()/close() (no implicit open in ctor).
- One-time importer now uses a direct async connection (no pool), and uses positional
  placeholders (no named placeholders with spaces).
- Added Windows event loop policy for CLI stability.
- Same cleaned design as before: no XLSX at runtime; only official schema via Postgres.

ENV:
  OPENAI_API_KEY
  PG_CONN  (e.g. postgresql://user:pass@host:5432/mazaya_mem)

USAGE:
  # One-time import
  python mazaya_agent_pg_streaming.py --import_xlsx ./Active.xlsx

  # Interactive CLI (streaming)
  python mazaya_agent_pg_streaming.py --user-id hamza --thread-id t1 --stream
"""

import os
import re
import sys
import json
import argparse
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

# LLM
from langchain_openai import ChatOpenAI

# LangGraph
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_config, get_store
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langchain.chat_models import init_chat_model
import vertexai
from dotenv import load_dotenv

load_dotenv()

# LangMem tools
from langmem import create_manage_memory_tool, create_search_memory_tool

# Tools decorator
from langchain_core.tools import tool
from langchain_core.messages import AIMessageChunk, ToolMessage

# Psycopg (async)
import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# ---------- Windows event loop fix ----------
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION"),
)

# ---------- Constants ----------
OFFICIAL_XLSX_COLUMNS = [
    "OfferID",
    "Company Name",
    "Offer Name En",
    "Offer Name Ar",
    "Offer Details En",
    "Offer Details Ar",
    "Effective",
    "Discontinue",
    "Unlimited",
    "Country",
    "Cities",
    "Category En",
    "Category Ar",
    "Status",
]

TABLE_NAME = "mazaya_offers_official"


# ---------- Utils ----------
def _norm(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def _lower(s: Any) -> str:
    return _norm(s).lower()


def _split_tokens(txt: str) -> List[str]:
    if not txt:
        return []
    return [t.strip() for t in re.split(r"[;,/|،]+", txt) if t.strip()]


# ---------- Canonical projection ----------
def row_to_canonical(r: Dict[str, Any]) -> Dict[str, Any]:
    offer_id = _norm(r.get("offer_id"))
    provider = _norm(r.get("company_name"))
    name_en = _norm(r.get("offer_name_en"))
    name_ar = _norm(r.get("offer_name_ar"))
    det_en = _norm(r.get("offer_details_en"))
    det_ar = _norm(r.get("offer_details_ar"))
    eff = _norm(r.get("effective"))
    disc = _norm(r.get("discontinue"))
    unlimited = _norm(r.get("unlimited"))
    country = _norm(r.get("country"))
    cities = _norm(r.get("cities"))
    cat_en = _norm(r.get("category_en"))
    cat_ar = _norm(r.get("category_ar"))

    rec_id = f"MZO-{offer_id or r.get('pk', '')}".strip("-")

    title = name_en or name_ar or provider or "(no title)"
    description = det_en if det_en else det_ar

    dur = []
    if eff:
        dur.append(f"effective: {eff}")
    if disc:
        dur.append(f"discontinue: {disc}")
    if unlimited and unlimited.lower() in ("yes", "true", "unlimited", "y", "1"):
        dur.append("unlimited")
    duration = " | ".join(dur)

    location = (cities or country).lower()

    tags_set = set()
    for t in _split_tokens(cat_en) + _split_tokens(cat_ar):
        tags_set.add(t.lower())
    for city in _split_tokens(cities):
        tags_set.add(city.lower())
    name_hint = (name_en or name_ar).lower() if (name_en or name_ar) else ""
    if "women" in name_hint or "نسائي" in name_hint:
        tags_set.add("women")
    if "family" in name_hint or "عائلة" in name_hint or "أُسرة" in name_hint:
        tags_set.add("family")

    return {
        "id": rec_id,
        "title": title,
        "category": (cat_en or cat_ar).lower(),
        "provider": provider,
        "description": description or "",
        "location": location,
        "duration": duration,
        "contact": "",
        "website": "",
        "url": "",
        "tags": ", ".join(sorted(tags_set)),
        "_cities_raw": cities,
        "_country_raw": country,
    }


# ---------- Benefits Repository ----------
class BenefitsRepo:
    """
    Async repository using a Postgres connection pool.

    Table:
      CREATE TABLE IF NOT EXISTS mazaya_offers_official (
        offer_id TEXT PRIMARY KEY,
        company_name TEXT,
        offer_name_en TEXT, offer_name_ar TEXT,
        offer_details_en TEXT, offer_details_ar TEXT,
        effective TEXT, discontinue TEXT,
        unlimited TEXT,
        country TEXT, cities TEXT,
        category_en TEXT, category_ar TEXT,
        status TEXT
      );
    """

    def __init__(self, dsn: str):
        self.dsn = dsn
        self.pool: Optional[AsyncConnectionPool] = None

    async def open(self):
        if self.pool is None:
            # Do NOT open in constructor. Open explicitly here.
            self.pool = AsyncConnectionPool(
                conninfo=self.dsn,
                min_size=1,
                max_size=8,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
        if self.pool.closed:  # psycopg ≥3.2 exposes .closed
            await self.pool.open()
        else:
            await self.pool.open()

    async def close(self):
        if self.pool is not None:
            try:
                await self.pool.close()
            except Exception:
                pass

    async def import_official_xlsx(self, path: str):
        """
        One-time importer: direct async connection (no pool).
        Uses positional placeholders to avoid issues with column names containing spaces.
        """
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        cols = {c.strip() for c in df.columns}
        if not set(OFFICIAL_XLSX_COLUMNS).issubset(cols):
            raise ValueError(
                "Input schema not recognized. Expected official Mazaya columns."
            )

        # Normalize dataframe column access to exact names
        async with await psycopg.AsyncConnection.connect(self.dsn) as con:
            async with con.cursor() as cur:
                await cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                        offer_id TEXT PRIMARY KEY,
                        company_name TEXT,
                        offer_name_en TEXT, offer_name_ar TEXT,
                        offer_details_en TEXT, offer_details_ar TEXT,
                        effective TEXT, discontinue TEXT,
                        unlimited TEXT,
                        country TEXT, cities TEXT,
                        category_en TEXT, category_ar TEXT,
                        status TEXT
                    );
                """
                )
                # Upsert each row
                for _, r in df.iterrows():
                    offer_id = r.get("OfferID")
                    if offer_id is None or (
                        isinstance(offer_id, float) and pd.isna(offer_id)
                    ):
                        # Skip rows without a stable OfferID
                        continue
                    values = (
                        r.get("OfferID"),
                        r.get("Company Name"),
                        r.get("Offer Name En"),
                        r.get("Offer Name Ar"),
                        r.get("Offer Details En"),
                        r.get("Offer Details Ar"),
                        r.get("Effective"),
                        r.get("Discontinue"),
                        r.get("Unlimited"),
                        r.get("Country"),
                        r.get("Cities"),
                        r.get("Category En"),
                        r.get("Category Ar"),
                        r.get("Status"),
                    )
                    await cur.execute(
                        f"""
                        INSERT INTO {TABLE_NAME} (
                          offer_id, company_name, offer_name_en, offer_name_ar,
                          offer_details_en, offer_details_ar, effective, discontinue,
                          unlimited, country, cities, category_en, category_ar, status
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (offer_id) DO UPDATE SET
                          company_name=EXCLUDED.company_name,
                          offer_name_en=EXCLUDED.offer_name_en,
                          offer_name_ar=EXCLUDED.offer_name_ar,
                          offer_details_en=EXCLUDED.offer_details_en,
                          offer_details_ar=EXCLUDED.offer_details_ar,
                          effective=EXCLUDED.effective,
                          discontinue=EXCLUDED.discontinue,
                          unlimited=EXCLUDED.unlimited,
                          country=EXCLUDED.country,
                          cities=EXCLUDED.cities,
                          category_en=EXCLUDED.category_en,
                          category_ar=EXCLUDED.category_ar,
                          status=EXCLUDED.status;
                        """,
                        values,
                    )
            await con.commit()

    async def _ensure_open(self):
        if self.pool is None or self.pool.closed:
            await self.open()

    async def _fetch_rows(
        self, where_sql: str = "", args: Tuple[Any, ...] = (), limit: int = 50
    ) -> List[Dict[str, Any]]:
        await self._ensure_open()
        sql = f"""
            SELECT
                offer_id, company_name, offer_name_en, offer_name_ar,
                offer_details_en, offer_details_ar, effective, discontinue,
                unlimited, country, cities, category_en, category_ar, status
            FROM {TABLE_NAME}
            WHERE (status IS NULL OR lower(status) = 'active')
            {('AND ' + where_sql) if where_sql else ''}
            LIMIT {int(limit)};
        """
        async with self.pool.connection() as con:
            async with con.cursor() as cur:
                await cur.execute(sql, args)
                rows = await cur.fetchall()
        return [row_to_canonical(r) for r in rows]

    async def search(
        self, query: str = "", filters: Optional[Dict[str, Any]] = None, limit: int = 12
    ) -> List[Dict[str, Any]]:
        filters = filters or {}
        args: List[Any] = []
        parts: List[str] = []

        if query:
            like = f"%{query}%"
            parts.append(
                """
                (
                    offer_name_en ILIKE %s OR offer_name_ar ILIKE %s OR
                    offer_details_en ILIKE %s OR offer_details_ar ILIKE %s OR
                    company_name ILIKE %s OR category_en ILIKE %s OR category_ar ILIKE %s OR
                    cities ILIKE %s OR country ILIKE %s
                )
            """
            )
            args.extend([like, like, like, like, like, like, like, like, like])

        f_cat = _lower(filters.get("category", ""))
        f_loc = _lower(filters.get("location", ""))
        f_tag = _lower(filters.get("tag", ""))
        f_prov = _lower(filters.get("provider_contains", ""))
        f_city = _lower(filters.get("city", ""))

        if f_cat:
            parts.append("(lower(category_en) = %s OR lower(category_ar) = %s)")
            args.extend([f_cat, f_cat])
        if f_loc:
            parts.append("(lower(country) = %s OR lower(cities) ILIKE %s)")
            args.extend([f_loc, f"%{f_loc}%"])
        if f_city:
            parts.append("lower(cities) ILIKE %s")
            args.append(f"%{f_city}%")
        if f_prov:
            parts.append("lower(company_name) ILIKE %s")
            args.append(f"%{f_prov}%")
        if f_tag:
            parts.append(
                "(lower(category_en) ILIKE %s OR lower(category_ar) ILIKE %s OR lower(cities) ILIKE %s)"
            )
            args.extend([f"%{f_tag}%", f"%{f_tag}%", f"%{f_tag}%"])

        base = await self._fetch_rows(
            " AND ".join([p.strip() for p in parts if p.strip()]),
            tuple(args),
            limit=100,
        )

        q = _lower(query)

        def score(rec: Dict[str, Any]) -> int:
            s = 0
            hay = " ".join(
                [
                    rec.get("title", ""),
                    rec.get("description", ""),
                    rec.get("category", ""),
                    rec.get("provider", ""),
                    rec.get("tags", ""),
                    rec.get("location", ""),
                ]
            ).lower()
            if q:
                for tok in [t for t in q.split() if t]:
                    if tok in hay:
                        s += 1
            if f_city and f_city in (rec.get("_cities_raw", "") or "").lower():
                s += 1
            if f_cat and f_cat == rec.get("category", ""):
                s += 2
            if f_prov and f_prov in (rec.get("provider", "") or "").lower():
                s += 1
            if f_tag and f_tag in rec.get("tags", ""):
                s += 2
            return s

        base.sort(key=lambda r: (-score(r), r.get("title", "")))
        return base[:limit]

    async def recommend(
        self, preferences: Dict[str, Any], limit: int = 8
    ) -> List[Dict[str, Any]]:
        filters: Dict[str, Any] = {}
        if preferences.get("city"):
            filters["city"] = preferences["city"]
        if preferences.get("location"):
            filters["location"] = preferences["location"]
        candidates = await self.search(
            query="", filters=filters, limit=max(60, limit * 6)
        )

        pref_tags = [_lower(t) for t in preferences.get("tags", [])]
        pref_cats = [_lower(c) for c in preferences.get("categories", [])]
        pref_loc = _lower(preferences.get("location", ""))
        pref_city = _lower(preferences.get("city", ""))
        disliked = [_lower(d) for d in preferences.get("disliked", [])]
        keywords = [_lower(k) for k in preferences.get("keywords", [])]

        def score(rec: Dict[str, Any]) -> int:
            s = 0
            tags = rec.get("tags", "")
            hay = " ".join([rec.get("title", ""), rec.get("description", "")]).lower()
            if pref_tags and any(t in tags for t in pref_tags):
                s += 3
            if pref_cats and rec.get("category", "") in pref_cats:
                s += 2
            if pref_loc and pref_loc in rec.get("location", ""):
                s += 1
            if pref_city and pref_city in (rec.get("_cities_raw", "") or "").lower():
                s += 1
            if disliked and any(d in hay for d in disliked):
                s -= 2
            for k in keywords:
                if k in hay:
                    s += 1
            return s

        candidates.sort(key=lambda r: (-score(r), r.get("title", "")))
        return candidates[:limit]


# ---------- Long-term memory schema ----------
class UserTraits(BaseModel):
    """Structured profile persisted long-term (via LangMem)."""

    name: Optional[str] = None
    role: Optional[str] = None
    department: Optional[str] = None
    location: Optional[str] = None
    family_status: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    recent_positive: List[str] = Field(default_factory=list)
    recent_negative: List[str] = Field(default_factory=list)


# ---------- Tools (async, explicit docstrings) ----------
_REPO: Optional[BenefitsRepo] = None


@tool("search_and_recommend_benefits", return_direct=False)
async def search_and_recommend_benefits(
    query: str = "", filters_json: str = "", limit: int = 12
) -> str:
    """
    Search Mazaya benefits (official catalog in Postgres).

    Use this when the user specifies cities/categories/providers or free-text.
    Filters JSON supports: {"category","location","tag","provider_contains","city"}.
    Prefer `recommend_benefits` for generic "suggest something" requests.

    Returns: JSON -> {"results":[...], "used_filters":{...}, "query":"..."}
    """
    if _REPO is None:
        return json.dumps({"error": "Benefits repository not initialized"})
    try:
        filters = json.loads(filters_json) if filters_json else {}
    except Exception:
        filters = {}
    rows = await _REPO.search(query=query or "", filters=filters, limit=limit)
    return json.dumps(
        {"results": rows, "used_filters": filters, "query": query or ""},
        ensure_ascii=False,
        indent=2,
    )


@tool("recommend_benefits", return_direct=False)
async def recommend_benefits(preferences_json: str = "", limit: int = 8) -> str:
    """
    Recommend Mazaya benefits using the user's long-term preferences.

    AGENT BEHAVIOR:
    - For generic requests ("show me some benefits"), first call the LangMem search
      tool to fetch ("memories","{langgraph_user_id}","profile"), extract `.preferences`,
      then pass them here.

    Args: preferences_json like {"tags":[],"categories":[],"keywords":[],"disliked":[],"city":"","location":""}
    Returns: JSON -> {"results":[...], "used_prefs":{...}}
    """
    if _REPO is None:
        return json.dumps({"error": "Benefits repository not initialized"})
    prefs: Dict[str, Any] = {}
    try:
        prefs = json.loads(preferences_json) if preferences_json else {}
    except Exception:
        prefs = {}
    rows = await _REPO.recommend(prefs, limit=limit)
    return json.dumps(
        {"results": rows, "used_prefs": prefs}, ensure_ascii=False, indent=2
    )


@tool("remember_traits", return_direct=False)
async def remember_traits(json_payload: str) -> str:
    """
    Persist stable traits/preferences under ("memories",{langgraph_user_id},"profile").
    Payload should match (partial) UserTraits. Example:
      {"location":"jeddah","preferences":{"tags":["family","fitness"],"city":"jeddah"}}
    """
    cfg = get_config()
    store = get_store()
    try:
        data = json.loads(json_payload)
    except Exception as e:
        return f"Invalid JSON: {e}"
    ns = ("memories", cfg["configurable"]["langgraph_user_id"], "profile")
    current = await store.aget(ns, "profile")
    merged = {}
    if current and isinstance(current.value, dict) and "content" in current.value:
        merged = current.value["content"]
    elif current and isinstance(current.value, dict):
        merged = current.value
    merged.update(data)
    await store.aput(
        ns,
        "profile",
        {"content": merged},
        index={"text": json.dumps(merged, ensure_ascii=False)},
    )
    return "Saved."


# ---------- Prompt ----------
def system_prompt_with_memories(state: Dict[str, Any]):
    cfg = get_config()
    store = get_store()
    memories_text = ""
    try:
        ns = ("memories", cfg["configurable"]["langgraph_user_id"], "profile")
        doc = store.get(ns, "profile")
        if doc:
            content = doc.value.get("content", doc.value)
            memories_text = json.dumps(content, ensure_ascii=False)
    except Exception:
        memories_text = ""

    system_msg = f"""
        You are a helpful Employee Benefits Assistant (Mazaya Agent). Your goal is to help employees discover the best offers and discounts available in the Mazaya program.

        ### Instructions
        1. **Greeting**
        - Greet the user once at the start of the session:
            "Hello! I'm your Mazaya Employee Benefits Assistant. I can help you explore amazing discounts and offers."
        - Do NOT repeat greetings in later responses.

        2. **Personalization Onboarding**
        - At the start, check for user preferences in memory.
        - If no preferences exist, ask if they want to personalize.
        - If yes, ask 2-3 short questions:
            1. City/country
            2. Categories of interest
            3. Optional: short-term vs long-term deals
        - If preferences already exist in memory:
            - Do NOT list them back explicitly.
            - Use them implicitly in responses (e.g. if user is in Riyadh, say "I can help you explore amazing discounts and offers in Riyadh").
        
        3. **Mazaya Offer Categories**
        ["Restaurants", "Hotels", "Car Rentals", "Furniture", "Automotive Sales", "Maintenance", "Education", "Fitness", "Electronics", "Medical & Health", "Entertainment", "Fashion & Apparel", "Women's Care", "Financial Offers", "General Services", "Real Estate"]
            
        4. **Responding to Queries**
        - For Mazaya offers, call `search_and_recommend_benefits` with appropriate filters.
        - Do not invent offers.
        - Always require at least:
            - **Category** (e.g., Restaurants, Hotels, Fitness, etc.)
            - And **City or Country**
        - If the user only provides a city:
            - Check if the data has a matching city.
            - If the data instead has **"All KSA"** for that offer:
                - Treat it as valid and respond with those offers (but make it clear to the user).
                - Example:  
                    "I couldn't find offers specific to Riyadh, but here are offers available across **all of Saudi Arabia**."
        - Ask clarifying questions if the request is vague.
        - Show top 3-5 offers based on highest discount and user preferences, formatted as a bulleted list with:
            - Offer Name
            - Offer Details (**highlight discounts in bold**)
            - Location
            - Duration (either start - end date or Unlimited)
            - Status

        - If no offers are available, ask if they want to search another category or location.

        5. **Memory**
        - Save user preferences, personal info, and interests to long-term memory.
        - **Whenever the user shares a new category, city, country, or preference**:
            - Add it to long-term memory.
            - Do not delete or overwrite earlier preferences unless the user explicitly says to update/remove them.
            - Treat long-term memory as a growing profile of the user's interests.

        6. **Communication Guidelines**
        - Be concise, professional, and friendly.
        - Ask at most 1-2 short questions at a time.
        - After a tool call, provide a final summarized answer and stop reasoning further.

        ### Goal
        Make it easy and enjoyable for employees to discover the most relevant offers, even if they don't know exactly what to ask for.
        
        <LONG_TERM_MEMORY_JSON>
        {memories_text}
        </LONG_TERM_MEMORY_JSON>
        """

    return [{"role": "system", "content": system_msg}, *state["messages"]]


# ---------- Build agent ----------
async def build_agent(
    pg_conn: str,
    repo: BenefitsRepo,
    checkpointer: AsyncPostgresSaver,
    store: AsyncPostgresStore,
):
    global _REPO
    _REPO = repo

    # await store.setup()
    # await checkpointer.setup()

    manage_memory_tool = create_manage_memory_tool(
        namespace=("memories", "{langgraph_user_id}", "profile"), schema=UserTraits
    )
    search_memory_tool = create_search_memory_tool(
        namespace=("memories", "{langgraph_user_id}")
    )

    # llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3, streaming=True)
    llm = init_chat_model(
        "gemini-2.5-flash",
        model_provider="google_genai",
        temperature=0.2,
    )

    agent = create_react_agent(
        llm,
        tools=[
            search_and_recommend_benefits,
            create_manage_memory_tool(
                namespace=("memories", "{langgraph_user_id}", "profile"),
                # schema=UserTraits,
            ),
            create_search_memory_tool(namespace=("memories", "{langgraph_user_id}")),
        ],
        prompt=system_prompt_with_memories,
        store=store,
        checkpointer=checkpointer,
    )
    return agent


# ---------- CLI ----------
async def run_cli(pg_conn: str, user_id: str, thread_id: str, stream: bool):
    repo = BenefitsRepo(pg_conn)
    await repo.open()

    store_cm = AsyncPostgresStore.from_conn_string(
        pg_conn,
        # index={"dims": 1536, "embed": "openai:text-embedding-3-small"},
        index={"dims": 768, "embed": "google_vertexai:text-embedding-004"},
    )
    ckpt_cm = AsyncPostgresSaver.from_conn_string(pg_conn)

    store = await store_cm.__aenter__()
    checkpointer = await ckpt_cm.__aenter__()

    agent = await build_agent(pg_conn, repo, checkpointer, store)

    print("\nMazaya Benefits Agent (async streaming). Ctrl+C to exit.\n")
    try:
        while True:
            try:
                user = await asyncio.to_thread(input, "You: ")
            except KeyboardInterrupt:
                break
            user = (user or "").strip()
            if not user:
                continue

            config = {
                "configurable": {"langgraph_user_id": user_id, "thread_id": thread_id}
            }
            print("\nAgent: ", end="", flush=True)

            async for msg_chunk, meta in agent.astream(
                {"messages": [{"role": "user", "content": user}]},
                config=config,
                stream_mode="messages",
            ):
                if meta.get("langgraph_node") != "agent":
                    continue
                text = getattr(msg_chunk, "content", "")
                if isinstance(text, str) and text:
                    print(text, end="", flush=True)
                elif isinstance(text, list):
                    for part in text:
                        if isinstance(part, dict) and "text" in part:
                            print(part["text"], end="", flush=True)
            print("")
    finally:
        await repo.close()
        await ckpt_cm.__aexit__(None, None, None)
        await store_cm.__aexit__(None, None, None)


# ---------- Main ----------
async def _amain():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-id", default="anon")
    parser.add_argument("--thread-id", default="default")
    parser.add_argument(
        "--pg_conn",
        default=os.getenv(
            "PG_CONN",
        ),
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--import_xlsx", help="Path to official Mazaya XLSX to import, then exit"
    )
    args = parser.parse_args()

    if args.import_xlsx:
        repo = BenefitsRepo(args.pg_conn)
        # direct connection used inside; no need to open pool here
        await repo.import_official_xlsx(args.import_xlsx)
        print(f"Imported '{args.import_xlsx}' into {TABLE_NAME}.")
        return

    await run_cli(args.pg_conn, args.user_id, args.thread_id, args.stream)


if __name__ == "__main__":
    asyncio.run(_amain())
