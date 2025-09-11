#!/usr/bin/env python3
"""
Chainlit UI for the Mazaya Benefits Agent — Postgres-only, async.
"""

import sys
import os
import uuid
import json
import traceback
import asyncio
import chainlit as cl

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_core.messages import ToolMessage, AIMessageChunk
from langgraph.store.postgres.aio import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from mazaya_agent_pg_streaming import build_agent, BenefitsRepo

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

@cl.on_chat_start
async def on_chat_start():
    try:
        pg_conn  = os.getenv("PG_CONN", "postgresql://postgres:%254~K%5D%3BUAJCjL%7D%3CA%3B@34.166.107.36:5432/mazayamem?sslmode=require")
        maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
        map_id   = os.getenv("GOOGLE_MAP_ID", "map")

        app_user = cl.user_session.get("user")
        langgraph_user_id = app_user.identifier if app_user else "anon"
        thread_id = cl.user_session.get("id") or str(uuid.uuid4())

        store_cm = AsyncPostgresStore.from_conn_string(
            pg_conn,
            index={"dims": 1536, "embed": "openai:text-embedding-3-small"}
        )
        store = await store_cm.__aenter__()

        ckpt_cm = AsyncPostgresSaver.from_conn_string(pg_conn)
        checkpointer = await ckpt_cm.__aenter__()

        repo = BenefitsRepo(pg_conn)
        await repo.open()  # <-- important: explicitly open the pool
        cl.user_session.set("repo", repo)

        agent = await build_agent(pg_conn, repo, checkpointer, store)
        cl.user_session.set("agent", agent)
        cl.user_session.set("config_base", {"configurable": {"langgraph_user_id": langgraph_user_id, "thread_id": thread_id}})
        cl.user_session.set("store_cm", store_cm)
        cl.user_session.set("ckpt_cm", ckpt_cm)

        await cl.Message(
            content="Hi! 👋 I’m your **Mazaya Benefits Agent**.\nAsk about offers by city/category/provider, or say *recommend me something* to use your saved preferences.",
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

    open_steps: dict[str, tuple[cl.Step, cl.Step]] = {}
    arg_buffers: dict[str, str] = {}

    maps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")
    map_id = os.getenv("GOOGLE_MAP_ID", "map")
    pending_places: list[dict] | None = None

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

    async with cl.Step(name="Thinking…", type="llm") as think:
        think.input = message.content
        try:
            async for msg_chunk, meta in agent.astream(
                {"messages": [{"role": "user", "content": message.content}]},
                config=config,
                stream_mode="messages",
            ):
                step_no = meta.get("langgraph_step", 0)

                if isinstance(msg_chunk, AIMessageChunk):
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

                    for tcc in (getattr(msg_chunk, "tool_call_chunks", []) or []):
                        key = _key_for_chunk(tcc, step_no)
                        arg_buffers[key] = arg_buffers.get(key, "") + (tcc.get("args") or "")
                        tc_id = tcc.get("id")
                        if tc_id and tc_id in open_steps:
                            _, step = open_steps[tc_id]
                            step.input = arg_buffers[key]

                    text = getattr(msg_chunk, "content", "")
                    if isinstance(text, str) and text:
                        await ai_msg.stream_token(text)
                    elif isinstance(text, list):
                        for part in text:
                            if isinstance(part, dict) and "text" in part and part["text"]:
                                await ai_msg.stream_token(part["text"])

                elif isinstance(msg_chunk, ToolMessage) or getattr(msg_chunk, "type", None) == "tool":
                    tool_name = getattr(msg_chunk, "name", None)
                    tc_id = getattr(msg_chunk, "tool_call_id", None) or getattr(msg_chunk, "id", None)
                    out = getattr(msg_chunk, "content", "")

                    if tool_name in ("search_and_recommend_benefits", "recommend_benefits") and maps_key and pending_places is None:
                        try:
                            payload = json.loads(out) if isinstance(out, str) else out
                            rows = payload.get("results", []) if isinstance(payload, dict) else []
                            queries = _build_places_queries(rows)
                            if queries:
                                pending_places = queries
                        except Exception as e:
                            print(f"[MAP] parse error: {e}")

                    await _close_tool_step(tc_id or "unknown", out or "")

        except Exception as e:
            await ai_msg.stream_token(f"\n\n[Generation error: {e}]")

        think.name = "Thought"
        think.output = "Response generated."
        await think.update()

    if maps_key and pending_places:
        map_msg = await cl.Message(content="", author="Saudia").send()
        await cl.CustomElement(
            name="MazayaPlaces",
            display="inline",
            props={
                "apiKey": maps_key,
                "mapId": map_id,
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
    repo = cl.user_session.get("repo")
    if repo:
        try:
            await repo.close()
        except Exception:
            pass
