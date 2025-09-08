#!/usr/bin/env python3
"""
Chainlit UI for the Mazaya Benefits Agent
- Clean token streaming
- Thinking + per-tool steps
- Async Postgres store + async checkpointer
- Sends a rich Google Places map/list element (MazayaPlaces) when search/recommend tools return rows

ENV:
  OPENAI_API_KEY
  PG_CONN
  MAZAYA_DATA
  CHECKPOINTER_BACKEND   -> sqlite|postgres
  CHECKPOINT_DB          -> sqlite file (default: mazaya_checkpoints.db)
  GOOGLE_MAPS_API_KEY    -> browser-safe key (restrict by HTTP referrer)
"""
import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import uuid
import json
import traceback
import chainlit as cl
from langchain_core.messages import ToolMessage, AIMessageChunk

# --- Async store / checkpointers ---
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _HAS_ASYNC_PG_CKPT = True
except Exception:
    _HAS_ASYNC_PG_CKPT = False

from mazaya_agent_pg_streaming import build_agent

# -------- Utilities --------

@cl.cache
def load_benefits_rows_cached(file_path: str, file_mtime: float) -> list[dict]:
    from mazaya_agent_pg_streaming import BenefitsIndex
    return BenefitsIndex.from_file(file_path).rows

def _pretty_json_text(value, limit: int = 1200) -> str:
    if isinstance(value, (dict, list)):
        s = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        s = str(value)
        try:
            parsed = json.loads(s)
            s = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return (s[:limit] + "…") if len(s) > limit else s

async def _open_store(pg_conn: str, index_cfg: dict | None):
    store_cm = AsyncPostgresStore.from_conn_string(pg_conn, index=index_cfg)
    store = await store_cm.__aenter__()
    # Store setup is first time only; skip
    # try:
    #     await store.setup()
    # except Exception as e:
    #     print(f"[WARN] store.setup(): {e}")
    return store_cm, store

async def _open_checkpointer(backend: str, pg_conn: str, sqlite_path: str):
    backend = (backend or "sqlite").lower()
    if backend == "postgres":
        if not _HAS_ASYNC_PG_CKPT:
            raise RuntimeError("AsyncPostgresSaver not available. Install langgraph-checkpoint-postgres.")
        ckpt_cm = AsyncPostgresSaver.from_conn_string(pg_conn)
        ckpt = await ckpt_cm.__aenter__()
        # Checkpointer setup is first time only; skip
        # try:
        #     await ckpt.setup()
        # except Exception as e:
        #     print(f"[WARN] checkpointer.setup(): {e}")
        return ckpt_cm, ckpt
    ckpt_cm = AsyncSqliteSaver.from_conn_string(sqlite_path)
    ckpt = await ckpt_cm.__aenter__()
    return ckpt_cm, ckpt

# -------- Map helpers --------

def _first_city(row: dict) -> str:
    # Prefer official column; else parse location field
    cities = (row.get("_cities_raw") or "").strip()
    if cities:
        # split on commas/|/؛/Arabic comma
        parts = [c.strip() for c in cities.replace("،", ",").split(",") if c.strip()]
        if parts:
            return parts[0]
    loc = (row.get("location") or "").lower()
    # quick guesses
    for c in ["riyadh", "jeddah", "dammam", "al khobar", "khobar", "makkah", "madinah", "mecca"]:
        if c in loc:
            return c.title()
    return ""

def _build_places_queries(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        # Prefer provider, else title
        provider = (r.get("provider") or "").strip()
        title = (r.get("title") or "").strip()
        category = (r.get("category") or "").strip()
        city = _first_city(r)
        url_hint = (r.get("url") or "").strip()
        out.append({
            "id": r.get("id"),
            "title": title,
            "provider": provider,
            "category": category,
            "city": city,
            "country": "Saudi Arabia",
            "url": url_hint
        })
    # keep it reasonable for UI
    return out[:10]

# -------- Chainlit lifecycle --------

@cl.on_chat_start
async def on_chat_start():
    try:
        data_path    = os.getenv("MAZAYA_DATA", "./Active.xlsx")
        pg_conn      = os.getenv("PG_CONN", "postgresql://postgres:postgres@localhost:5432/mazaya_mem")
        ckpt_backend = os.getenv("CHECKPOINTER_BACKEND", "sqlite")
        sqlite_file  = os.getenv("CHECKPOINT_DB", "mazaya_checkpoints.db")

        # Long-term id
        app_user = cl.user_session.get("user")
        langgraph_user_id = app_user.identifier if app_user else "anon"
        # Per chat thread
        thread_id = cl.user_session.get("id") or str(uuid.uuid4())

        # Vector index for store (set to None if you don't use pgvector)
        index_cfg = {
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }

        # Warm Excel via cache
        mtime = os.path.getmtime(data_path)
        rows = load_benefits_rows_cached(data_path, mtime)
        print(f"[INFO] Loaded {len(rows)} benefits rows from cache.")

        # Open resources
        store_cm, store = await _open_store(pg_conn, index_cfg)
        ckpt_cm, checkpointer = await _open_checkpointer(ckpt_backend, pg_conn, sqlite_file)

        # Build agent (builder accepts preloaded rows to skip re-parsing)
        agent = build_agent(file_path=data_path, checkpointer=checkpointer, store=store, preloaded_rows=rows)

        cl.user_session.set("agent", agent)
        cl.user_session.set("config_base", {
            "configurable": {"langgraph_user_id": langgraph_user_id, "thread_id": thread_id}
        })
        cl.user_session.set("store_cm", store_cm)
        cl.user_session.set("ckpt_cm", ckpt_cm)

        await cl.Message(
            content="Hi! 👋 I’m your **Mazaya Benefits Agent**.\nAsk about offers by city/category/provider, or tell me your preferences for recommendations.",
            author="Saudia",
        ).send()

    except Exception as e:
        await cl.Message(content=f"Init error: {e}").send()
        traceback.print_exc()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    return None

@cl.on_message
async def on_message(message: cl.Message):
    agent  = cl.user_session.get("agent")
    config = cl.user_session.get("config_base")
    if agent is None:
        await cl.Message(content="Agent not initialized. Please /restart the chat.").send()
        return

    ai_msg = cl.Message(content="", author="Saudia")

    # tool_call_id -> (ctxmgr, step)
    open_steps: dict[str, tuple[cl.Step, cl.Step]] = {}
    arg_buffers: dict[str, str] = {}

    maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    map_id = os.getenv("GOOGLE_MAP_ID", "map")  # <-- NEW: map element id
    pending_places: list[dict] | None = None  # <-- buffer here; render after the step

    def _pretty_json_text(value: str | dict | list, limit: int = 1200) -> str:
        if isinstance(value, (dict, list)):
            s = json.dumps(value, ensure_ascii=False, indent=2)
        else:
            s = str(value)
            try:
                parsed = json.loads(s)
                s = json.dumps(parsed, ensure_ascii=False, indent=2)
            except Exception:
                pass
        return (s[:limit] + "…") if len(s) > limit else s

    def _key_for_chunk(tcc: dict, step_no: int) -> str:
        return tcc.get("id") or f"idx:{tcc.get('index', 0)}:step:{step_no}"

    async def _open_tool_step(tc_id: str, name: str, seed_args: str = ""):
        if tc_id in open_steps: return
        cm = cl.Step(name=f"🔧 {name or 'tool'}", type="tool")
        step = await cm.__aenter__()
        step.input = _pretty_json_text(seed_args)
        open_steps[tc_id] = (cm, step)

    async def _close_tool_step(tc_id: str, output_text: str):
        ctx = open_steps.pop(tc_id, None)
        if not ctx:
            cm = cl.Step(name="🔧 tool", type="tool")
            step = await cm.__aenter__()
            step.input = arg_buffers.get(tc_id, "")
            step.output = _pretty_json_text(output_text)
            await cm.__aexit__(None, None, None)
            return
        cm, step = ctx
        step.output = _pretty_json_text(output_text)
        await cm.__aexit__(None, None, None)

    # ---- helpers for Places queries from benefits rows ----
    def _first_city(row: dict) -> str:
        cities = (row.get("_cities_raw") or "").strip()
        if cities:
            parts = [c.strip() for c in cities.replace("،", ",").split(",") if c.strip()]
            if parts: return parts[0]
        loc = (row.get("location") or "").lower()
        for c in ["riyadh", "jeddah", "dammam", "al khobar", "khobar", "makkah", "madinah", "mecca"]:
            if c in loc: return c.title()
        return ""

    def _build_places_queries(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            out.append({
                "id": r.get("id"),
                "title": (r.get("title") or "").strip(),
                "provider": (r.get("provider") or "").strip(),
                "category": (r.get("category") or "").strip(),
                "city": _first_city(r),
                "country": "Saudi Arabia",
                "url": (r.get("url") or "").strip()
            })
        return out[:10]

    # ---------- run the agent with a visible "Thinking…" step ----------
    async with cl.Step(name="Thinking…", type="llm") as think:
        think.input = message.content
        try:
            async for msg_chunk, meta in agent.astream(
                {"messages": [{"role": "user", "content": message.content}]},
                config=config,
                stream_mode="messages",
            ):
                step_no = meta.get("langgraph_step", 0)

                # LLM chunks
                if isinstance(msg_chunk, AIMessageChunk):
                    # open tool steps when id+name appear
                    for tc in (getattr(msg_chunk, "tool_calls", []) or []):
                        name = tc.get("name") or tc.get("function", {}).get("name")
                        tc_id = tc.get("id")
                        idx   = tc.get("index", None)
                        if not (name and tc_id): continue
                        if idx is not None:
                            idx_key = f"idx:{idx}:step:{step_no}"
                            if idx_key in arg_buffers:
                                arg_buffers[tc_id] = arg_buffers.get(tc_id, "") + arg_buffers.pop(idx_key)
                        seed = arg_buffers.get(tc_id, "")
                        await _open_tool_step(tc_id, name, seed_args=seed)

                    # stream tool args into the step
                    for tcc in (getattr(msg_chunk, "tool_call_chunks", []) or []):
                        key = _key_for_chunk(tcc, step_no)
                        arg_buffers[key] = arg_buffers.get(key, "") + (tcc.get("args") or "")
                        tc_id = tcc.get("id")
                        if tc_id and tc_id in open_steps:
                            _, step = open_steps[tc_id]
                            step.input = arg_buffers[key]

                    # stream assistant text
                    text = getattr(msg_chunk, "content", "")
                    if isinstance(text, str) and text:
                        await ai_msg.stream_token(text)
                    elif isinstance(text, list):
                        for part in text:
                            if isinstance(part, dict) and "text" in part and part["text"]:
                                await ai_msg.stream_token(part["text"])

                # Tool results
                elif isinstance(msg_chunk, ToolMessage) or getattr(msg_chunk, "type", None) == "tool":
                    tool_name = getattr(msg_chunk, "name", None)
                    tc_id = getattr(msg_chunk, "tool_call_id", None) or getattr(msg_chunk, "id", None)
                    out = getattr(msg_chunk, "content", "")

                    # If this is a benefits tool, PREPARE places data (do NOT send here).
                    if tool_name in ("search_benefits", "recommend_benefits") and maps_key and pending_places is None:
                        try:
                            payload = json.loads(out) if isinstance(out, str) else out
                            rows = payload.get("results", []) if isinstance(payload, dict) else []
                            queries = _build_places_queries(rows)
                            if queries:
                                pending_places = queries  # <-- buffer for later (outside the step)
                        except Exception as e:
                            print(f"[MAP] parse error: {e}")

                    await _close_tool_step(tc_id or "unknown", out or "")

        except Exception as e:
            await ai_msg.stream_token(f"\n\n[Generation error: {e}]")

        think.name = "Thought"
        think.output = "Response generated."
        await think.update()

    # ---------- OUTSIDE the step: render the map, then the assistant message ----------
    if maps_key and pending_places:
        map_msg = await cl.Message(content="", author="Saudia").send()
        await cl.CustomElement(
            name="MazayaPlaces",
            display="inline",  # inline | side | page
            props={
                "apiKey": maps_key,
                "mapId": map_id,  # <<< NEW
                "queries": pending_places,
                "listFirst": True,
                "region": "SA",
                "language": "en",
            },
        ).send(for_id=map_msg.id)

    await ai_msg.send()



@cl.on_chat_end
async def on_chat_end():
    ckpt_cm = cl.user_session.get("ckpt_cm")
    if ckpt_cm:
        try:
            await ckpt_cm.__aexit__(None, None, None)
        except Exception:
            pass

    store_cm = cl.user_session.get("store_cm")
    if store_cm:
        try:
            await store_cm.__aexit__(None, None, None)
        except Exception:
            pass
