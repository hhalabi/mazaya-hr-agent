#!/usr/bin/env python3
"""
Mazaya Benefits Agent (LangGraph + LangMem + PostgresStore)

Supports the official Mazaya Excel schema:
  OfferID, Company Name, Offer Name En, Offer Name Ar, Offer Details En, Offer Details Ar,
  Effective, Discontinue, Unlimited, Country, Cities, Category En, Category Ar, Status

Also supports the previous internal sheet:
  Category, CardName, Title, Offer, Location, Duration, Contact, Website, Offer_URL

Run:
  export OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
  export PG_CONN="postgresql://postgres:postgres@localhost:5432/mazaya_mem"
  python mazaya_agent_pg.py --csv_or_xlsx ./Active.xlsx --user-id hamza --thread-id t1
"""

import os
import json
import argparse
import re
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# ---- I/O
import pandas as pd

# ---- LLM (OpenAI)
from langchain_openai import ChatOpenAI

# ---- LangGraph core
from langgraph.prebuilt import create_react_agent
from langgraph.config import get_config, get_store
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

# Prefer SQLite checkpointer for persistence if available
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # pip install langgraph-checkpoint-sqlite
    _HAS_SQLITE = True
except Exception:
    _HAS_SQLITE = False

# ---- Tools & memory helpers
from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool


# =========================
# Helpers
# =========================

def _norm(s: Any) -> str:
    return (str(s).strip() if s is not None else "").replace("\n", " ").replace("  ", " ")

def _lower(s: Any) -> str:
    return _norm(s).lower()

def _split_tokens(txt: str) -> List[str]:
    if not txt:
        return []
    return [t.strip() for t in re.split(r"[;,/|،]+", txt) if t.strip()]

def _join_locs(country: str, cities: str) -> str:
    country = _norm(country)
    cities = _norm(cities)
    if cities:
        return cities.lower()
    return country.lower()

# Back-compat location parser (kept for older sheet)
def _extract_city_from_location(loc: str) -> str:
    loc = _norm(loc)
    m = re.search(r"\(([^)]+)\)", loc)
    if m:
        return _lower(m.group(1))
    alias = _lower(loc)
    alias = alias.replace("saudi arabia", "ksa").replace("kingdom-wide", "ksa-wide").strip()
    return alias


# =========================
# Domain: Mazaya Benefits
# =========================

CANONICAL_FIELDS = [
    "id", "title", "category", "provider", "description",
    "location", "duration", "contact", "website", "url", "tags"
]

# Old sheet columns (kept for compatibility)
OLD_XLSX_COLUMNS = [
    "Category", "CardName", "Title", "Offer", "Location", "Duration", "Contact", "Website", "Offer_URL"
]

# Official Mazaya columns (new dataset)
OFFICIAL_XLSX_COLUMNS = [
    "OfferID", "Company Name", "Offer Name En", "Offer Name Ar",
    "Offer Details En", "Offer Details Ar", "Effective", "Discontinue",
    "Unlimited", "Country", "Cities", "Category En", "Category Ar", "Status"
]

class BenefitsIndex:
    """
    In-memory index over Excel/CSV.
    - Auto-detects *official* Mazaya schema vs the older internal table.
    - Normalizes to canonical fields.
    - Search across English + Arabic (name/details/category) & provider, tags, location, duration.
    """
    def __init__(self, rows: List[Dict[str, str]]):
        self.rows = rows

    @staticmethod
    def _load_frame(path: str) -> pd.DataFrame:
        ext = path.lower().split(".")[-1]
        if ext in ("xlsx", "xls"):
            # openpyxl is required for .xlsx
            return pd.read_excel(path, sheet_name=0, engine="openpyxl")
        elif ext == "csv":
            return pd.read_csv(path, encoding="utf-8")
        else:
            raise ValueError("Unsupported file type. Please provide .xlsx/.xls or .csv")

    @staticmethod
    def from_file(path: str) -> "BenefitsIndex":
        df = BenefitsIndex._load_frame(path)

        cols = set(df.columns)
        if set(OFFICIAL_XLSX_COLUMNS).issubset(cols):
            rows = BenefitsIndex._normalize_official(df)
        elif set(OLD_XLSX_COLUMNS).issubset(cols):
            rows = BenefitsIndex._normalize_old(df)
        else:
            raise ValueError(
                "Input schema not recognized. Expected official Mazaya columns or the older internal columns."
            )
        return BenefitsIndex(rows)

    @staticmethod
    def _normalize_official(df: pd.DataFrame) -> List[Dict[str, str]]:
        # Filter to Active if provided
        if "Status" in df.columns:
            try:
                df = df[df["Status"].astype(str).str.lower().eq("active") | df["Status"].isna()]
            except Exception:
                pass

        rows: List[Dict[str, str]] = []
        for i, r in df.iterrows():
            offer_id = _norm(r.get("OfferID"))
            provider = _norm(r.get("Company Name"))
            name_en  = _norm(r.get("Offer Name En"))
            name_ar  = _norm(r.get("Offer Name Ar"))
            det_en   = _norm(r.get("Offer Details En"))
            det_ar   = _norm(r.get("Offer Details Ar"))
            eff      = _norm(r.get("Effective"))
            disc     = _norm(r.get("Discontinue"))
            unlimited= _norm(r.get("Unlimited"))
            country  = _norm(r.get("Country"))
            cities   = _norm(r.get("Cities"))
            cat_en   = _norm(r.get("Category En"))
            cat_ar   = _norm(r.get("Category Ar"))

            # ID
            if offer_id:
                rec_id = f"MZO-{offer_id}"
            else:
                slug_bits = f"{provider}-{name_en or name_ar or 'offer'}"
                slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug_bits).strip("-").lower()[:40]
                rec_id = f"MZO-{i+1:04d}-{slug}"

            # Title/Description
            title = name_en or name_ar or provider or "(no title)"
            # Merge EN/AR details for searchability; show EN first
            description = det_en
            if det_ar:
                description = f"{description} | {det_ar}" if description else det_ar

            # Duration string
            dur_bits = []
            if eff: dur_bits.append(f"effective: {eff}")
            if disc: dur_bits.append(f"discontinue: {disc}")
            if unlimited and unlimited.lower() in ("yes", "true", "unlimited", "y", "1"):
                dur_bits.append("unlimited")
            duration = " | ".join(dur_bits)

            # Location
            location = _join_locs(country, cities)

            # Tags: categories (EN/AR tokens) + city tokens
            tags_set = set()
            for t in _split_tokens(cat_en) + _split_tokens(cat_ar):
                tags_set.add(t.lower())
            for city in _split_tokens(cities):
                tags_set.add(city.lower())
            # Add simple hints from names
            name_hint = (name_en or name_ar).lower() if (name_en or name_ar) else ""
            if "women" in name_hint or "نسائي" in name_hint:
                tags_set.add("women")
            if "family" in name_hint or "عائلة" in name_hint or "أُسرة" in name_hint:
                tags_set.add("family")

            canonical = {
                "id": rec_id,
                "title": title,
                "category": (cat_en or cat_ar).lower(),
                "provider": provider,
                "description": description,
                "location": location,
                "duration": duration,
                "contact": "",     # not provided in official sheet
                "website": "",     # not provided in official sheet
                "url": "",         # not provided in official sheet
                "tags": ", ".join(sorted(tags_set)),
                # keep hidden fields to widen search (not exposed directly)
                "_name_ar": name_ar,
                "_details_ar": det_ar,
                "_cat_ar": cat_ar,
                "_cities_raw": cities,
                "_country_raw": country,
            }
            rows.append(canonical)
        return rows

    @staticmethod
    def _normalize_old(df: pd.DataFrame) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for i, r in df.iterrows():
            category  = _norm(r.get("Category"))
            provider  = _norm(r.get("CardName"))
            title     = _norm(r.get("Title"))
            offer     = _norm(r.get("Offer"))
            location  = _norm(r.get("Location"))
            duration  = _norm(r.get("Duration"))
            contact   = _norm(r.get("Contact"))
            website   = _norm(r.get("Website"))
            url       = _norm(r.get("Offer_URL"))

            slug_bits = f"{provider}-{title}" if provider or title else f"row-{i+1}"
            slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug_bits).strip("-").lower()[:40]
            rec_id = f"MZ{(i+1):04d}-{slug}" if slug else f"MZ{(i+1):04d}"

            tags = []
            if category:
                parts = [p.strip() for p in re.split(r"[/,|]+", category) if p.strip()]
                tags.extend([p.lower() for p in parts])
            city = _extract_city_from_location(location)
            if city and city not in tags:
                tags.append(city)

            canonical = {
                "id": rec_id,
                "title": title or provider or "(no title)",
                "category": category.lower(),
                "provider": provider,
                "description": offer,
                "location": location.lower(),
                "duration": duration,
                "contact": contact,
                "website": website,
                "url": url,
                "tags": ", ".join(sorted(set(tags))),
            }
            rows.append(canonical)
        return rows

    def _match_score(self, text: str, hay: str) -> int:
        score = 0
        for token in [t.strip() for t in text.split() if t.strip()]:
            if token in hay:
                score += 1
        return score

    def search(self, query: str = "", filters: Optional[Dict[str, Any]] = None, limit: int = 12) -> List[Dict[str, Any]]:
        """
        Keyword search + simple filters
          filters: {category, location, tag, provider_contains, city}
        """
        q = _lower(query or "")
        filters = filters or {}
        f_cat = _lower(filters.get("category", ""))
        f_loc = _lower(filters.get("location", ""))
        f_tag = _lower(filters.get("tag", ""))
        f_prov = _lower(filters.get("provider_contains", ""))
        f_city = _lower(filters.get("city", ""))

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for r in self.rows:
            hay = " ".join([
                r.get("title",""), r.get("description",""), r.get("category",""),
                r.get("provider",""), r.get("tags",""), r.get("location",""),
                r.get("duration",""), r.get("_name_ar",""), r.get("_details_ar",""), r.get("_cat_ar","")
            ]).lower()

            # hard filters
            if f_cat and f_cat not in r.get("category",""): continue
            if f_loc and f_loc not in r.get("location",""): continue
            if f_tag and f_tag not in r.get("tags",""): continue
            if f_city and f_city not in r.get("_cities_raw","").lower(): continue
            if f_prov and f_prov not in r.get("provider","").lower(): continue

            score = 0
            if q:
                score += self._match_score(q, hay)
            scored.append((score, r))

        scored.sort(key=lambda x: (-x[0], x[1].get("title","")))
        return [s[1] for s in scored[:limit]]

    def recommend(self, preferences: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
        """
        Scoring:
          +3 tag match
          +2 category match
          +1 location/city match
          -2 disliked keyword found
          + keywords matched in title/description (EN/AR)
        """
        pref_tags = [_lower(t) for t in preferences.get("tags", [])]
        pref_cats = [_lower(c) for c in preferences.get("categories", [])]
        pref_loc  = _lower(preferences.get("location", ""))
        pref_city = _lower(preferences.get("city", ""))
        disliked  = [_lower(d) for d in preferences.get("disliked", [])]
        keywords  = [_lower(k) for k in preferences.get("keywords", [])]

        scored = []
        for r in self.rows:
            score = 0
            tags = r.get("tags","")
            hay = " ".join([
                r.get("title",""),
                r.get("description",""),
                r.get("_details_ar","")
            ]).lower()

            if pref_tags and any(t in tags for t in pref_tags): score += 3
            if pref_cats and r.get("category","") in pref_cats: score += 2
            if pref_loc and pref_loc in r.get("location",""): score += 1
            if pref_city and pref_city in r.get("_cities_raw","").lower(): score += 1
            if disliked and any(d in hay for d in disliked): score -= 2
            for k in keywords:
                if k in hay:
                    score += 1

            scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], x[1].get("title","")))
        return [s[1] for s in scored[:limit]]


# =========================
# Long-term memory schema
# =========================

class UserTraits(BaseModel):
    """Structured profile persisted long-term."""
    name: Optional[str] = None
    role: Optional[str] = None               # "cabin crew", "ground ops", etc.
    department: Optional[str] = None
    location: Optional[str] = None           # city/country
    family_status: Optional[str] = None      # "single", "married", "kids"
    languages: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)  # "outdoorsy", "budget-conscious"
    preferences: Dict[str, Any] = Field(default_factory=dict)    # tags/categories/keywords/disliked/city
    recent_positive: List[str] = Field(default_factory=list)
    recent_negative: List[str] = Field(default_factory=list)


# ================
# Tools
# ================

_BENEFITS_INDEX: Optional[BenefitsIndex] = None

@tool("search_benefits", return_direct=False)
def search_benefits(query: str = "", filters_json: str = "", limit: int = 12) -> str:
    """
    Search benefits from XLSX/CSV-backed index.
    Filters (JSON): {category, location, tag, provider_contains, city}
    """
    if _BENEFITS_INDEX is None:
        return json.dumps({"error": "Benefits index not initialized"})
    try:
        filters = json.loads(filters_json) if filters_json else {}
    except Exception:
        filters = {}
    rows = _BENEFITS_INDEX.search(query=query, filters=filters, limit=limit)
    return json.dumps({"results": rows})

@tool("recommend_benefits", return_direct=False)
def recommend_benefits(preferences_json: str = "", limit: int = 8) -> str:
    """
    Recommend benefits based on preferences JSON:
      { tags:[], categories:[], location:"", city:"", disliked:[], keywords:[] }
    """
    if _BENEFITS_INDEX is None:
        return json.dumps({"error": "Benefits index not initialized"})
    try:
        prefs = json.loads(preferences_json) if preferences_json else {}
    except Exception:
        prefs = {}
    rows = _BENEFITS_INDEX.recommend(prefs, limit=limit)
    return json.dumps({"results": rows, "used_prefs": prefs})

# Optional deterministic memory save tool (kept; handy in demos/tests)
@tool("remember_traits", return_direct=False)
def remember_traits(json_payload: str) -> str:
    """
    Force-save stable traits/preferences. json_payload should match UserTraits fields (partial ok).
    Example:
      {"location":"jeddah","preferences":{"tags":["yoga","family"],"city":"jeddah"}}
    """
    cfg = get_config()
    store = get_store()
    try:
        data = json.loads(json_payload)
    except Exception as e:
        return f"Invalid JSON: {e}"
    ns = ("memories", cfg["configurable"]["langgraph_user_id"], "profile")
    current = store.get(ns, "profile")
    merged = {}
    if current and isinstance(current.value, dict) and "content" in current.value:
        merged = current.value["content"]
    elif current and isinstance(current.value, dict):
        merged = current.value
    # shallow merge
    merged.update(data)
    store.put(ns, "profile", {"content": merged}, index={"text": json.dumps(merged, ensure_ascii=False)})
    return "Saved."


# =========================
# Prompt (inject memories)
# =========================

def system_prompt_with_memories(state: Dict[str, Any]):
    cfg = get_config()
    store = get_store()

    # Load stored profile for this user-id
    memories_text = ""
    try:
        ns = ("memories", cfg["configurable"]["langgraph_user_id"], "profile")
        doc = store.get(ns, "profile")
        if doc:
            content = doc.value.get("content", doc.value)
            memories_text = json.dumps(content, ensure_ascii=False)
    except Exception:
        memories_text = ""

    rules = f"""
You are the "Mazaya Benefits Agent" for Saudia Airlines' employee benefits program (Mazaya).
Your job:
  1) Answer questions strictly from the structured benefits table via tools.
  2) Start friendly, gather context (role/department, location/city, family status, interests, budget/fitness/health/etc.). Load memories first to avoid repeating questions.
  3) Recommend the best benefits, explaining *why* briefly (match to user preferences).
  4) Maintain *long-term memory* of stable user traits/preferences using the memory tool(s).
  5) Use *short-term* thread memory naturally for conversation flow.

When you learn a new stable preference/trait, proactively call the memory tool to CREATE/UPDATE the user's profile.
If unsure, ask concise follow-ups. Keep answers clear and scoped to Mazaya.

<LONG_TERM_MEMORY_JSON>
{memories_text}
</LONG_TERM_MEMORY_JSON>
""".strip()

    return [{"role": "system", "content": rules}, *state["messages"]]


# =========================
# Build the agent
# =========================

def build_agent(file_path: str, checkpointer, store: PostgresStore):
    global _BENEFITS_INDEX
    _BENEFITS_INDEX = BenefitsIndex.from_file(file_path)

    # LangMem tools against persistent PostgresStore
    manage_memory_tool = create_manage_memory_tool(
        namespace=("memories", "{langgraph_user_id}", "profile"),
        schema=UserTraits,
        store=store,
    )
    search_memory_tool = create_search_memory_tool(
        namespace=("memories", "{langgraph_user_id}"),
        store=store,
    )

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0.3,
        # api_key=os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_KEY_HERE"),
    )

    agent = create_react_agent(
        llm,
        tools=[search_benefits, recommend_benefits, manage_memory_tool, search_memory_tool, remember_traits],
        prompt=system_prompt_with_memories,
        store=store,           # persistent long-term store
        checkpointer=checkpointer,  # per-thread short-term memory
    )
    return agent


# =========================
# CLI / example usage
# =========================

def run_cli_loop(agent, args):
    print("\nMazaya Benefits Agent ready. Type your question (Ctrl+C to exit).\n")
    while True:
        try:
            user = input("You: ").strip()
            if not user:
                continue
            config = {
                "configurable": {
                    "langgraph_user_id": args.user_id,  # long-term namespace key
                    "thread_id": args.thread_id,        # short-term (checkpoint) key
                }
            }
            out = agent.invoke({"messages": [{"role": "user", "content": user}]}, config=config)
            print(f"\nAgent: {out['messages'][-1].content}\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_or_xlsx", required=True, help="Path to Mazaya offers XLSX/CSV")
    parser.add_argument("--user-id", required=True, help="Stable user ID for long-term memory namespace")
    parser.add_argument("--thread-id", default="default", help="Conversation thread (short-term memory scope)")
    parser.add_argument("--pg-conn", default=os.getenv("PG_CONN", "postgresql://postgres:postgres@localhost:5432/mazaya_mem"))
    args = parser.parse_args()

    # ---- Persistent long-term memory store (Postgres) ----
    with PostgresStore.from_conn_string(
        args.pg_conn,
        index={"dims": 1536, "embed": "openai:text-embedding-3-small"}  # requires pgvector; set to None if not available
        # index=None
    ) as store:
        try:
            store.setup()
        except Exception as e:
            print(f"[WARN] store.setup() failed (vector index may be unavailable): {e}")

        # ---- Short-term memory (SQLite checkpointer) ----
        if _HAS_SQLITE:
            with SqliteSaver.from_conn_string("mazaya_checkpoints.db") as checkpointer:
                agent = build_agent(args.csv_or_xlsx, checkpointer=checkpointer, store=store)
                run_cli_loop(agent, args)
        else:
            checkpointer = MemorySaver()
            agent = build_agent(args.csv_or_xlsx, checkpointer=checkpointer, store=store)
            run_cli_loop(agent, args)

if __name__ == "__main__":
    main()
