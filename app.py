#!/usr/bin/env python3
"""
Chainlit UI for the Mazaya Benefits Agent (Async + Clean Streaming + Tool Steps)

- Streams ONLY final assistant text (filters out tool chatter)
- "Thinking…" step while the agent works
- Separate steps for each tool call (start with args, finish with result)
- Async Postgres store (long-term memory) + Async Sqlite/Postgres checkpointer (short-term)
- Works with your existing build_agent(file_path, checkpointer, store)

ENV:
  OPENAI_API_KEY         -> OpenAI key for gpt-4.1-mini
  PG_CONN                -> postgresql://postgres:postgres@localhost:5432/mazaya_mem (for both store & pg checkpointer)
  MAZAYA_DATA            -> /path/to/Active.xlsx
  CHECKPOINTER_BACKEND   -> "sqlite" (default) or "postgres"
  CHECKPOINT_DB          -> (sqlite) file, default "mazaya_checkpoints.db"
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

# --- Async store for long-term memory (Postgres) ---
from langgraph.store.postgres import AsyncPostgresStore

# --- Async checkpointers for short-term/thread memory ---
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
try:
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    _HAS_ASYNC_PG_CKPT = True
except Exception:
    _HAS_ASYNC_PG_CKPT = False

from mazaya_agent_pg_streaming import build_agent



# ---------- Helpers: open/close async resources ----------

@cl.cache  # 1h cache; adjust as you like
def load_benefits_rows_cached(file_path: str, file_mtime: float) -> list[dict]:
    """
    Parse/normalize the Mazaya Excel once, reuse across sessions.
    Caching key = (file_path, file_mtime), so new uploads break the cache.
    """
    from mazaya_agent_pg_streaming import BenefitsIndex
    return BenefitsIndex.from_file(file_path).rows

def _pretty_json_text(value: str | dict | list, limit: int = 1200) -> str:
    if isinstance(value, (dict, list)):
        s = json.dumps(value, ensure_ascii=False, indent=2)
    else:
        s = str(value)
        # If it's JSON text, parse and re-dump without escapes
        try:
            parsed = json.loads(s)
            s = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return (s[:limit] + "…") if len(s) > limit else s


async def _open_store(pg_conn: str, index_cfg: dict | None):
    """
    Open AsyncPostgresStore and run setup().
    Returns (ctxmgr, store) so we can __aexit__ later.
    """
    store_cm = AsyncPostgresStore.from_conn_string(pg_conn, index=index_cfg)
    store = await store_cm.__aenter__()
    try:
        await store.setup()
    except Exception as e:
        print(f"[WARN] store.setup() failed or already applied: {e}")
    return store_cm, store


async def _open_checkpointer(backend: str, pg_conn: str, sqlite_path: str):
    """
    Open an async checkpointer: sqlite (default) or postgres.
    Returns (ctxmgr, checkpointer).
    """
    backend = (backend or "sqlite").lower()
    if backend == "postgres":
        if not _HAS_ASYNC_PG_CKPT:
            raise RuntimeError("AsyncPostgresSaver not available. Install langgraph-checkpoint-postgres.")
        ckpt_cm = AsyncPostgresSaver.from_conn_string(pg_conn)
        ckpt = await ckpt_cm.__aenter__()
        try:
            await ckpt.setup()
        except Exception as e:
            print(f"[WARN] checkpointer.setup() (async) failed or not needed: {e}")
        return ckpt_cm, ckpt

    ckpt_cm = AsyncSqliteSaver.from_conn_string(sqlite_path)
    ckpt = await ckpt_cm.__aenter__()
    # AsyncSqliteSaver usually needs no explicit setup
    return ckpt_cm, ckpt


# ---------- Chainlit lifecycle ----------

@cl.on_chat_start
async def on_chat_start():
    try:
        data_path    = os.getenv("MAZAYA_DATA", "./Active.xlsx")
        pg_conn      = os.getenv("PG_CONN", "postgresql://postgres:postgres@localhost:5432/mazaya_mem")
        ckpt_backend = os.getenv("CHECKPOINTER_BACKEND", "sqlite")
        sqlite_file  = os.getenv("CHECKPOINT_DB", "mazaya_checkpoints.db")

        # Long-term user id (auth-aware if enabled)
        app_user = cl.user_session.get("user")
        langgraph_user_id = app_user.identifier if app_user else "anon"

        # Short-term thread id (per chat)
        thread_id = cl.user_session.get("id") or str(uuid.uuid4())

        # Enable pgvector if available; otherwise set to None
        index_cfg = {
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }
        
        mtime = os.path.getmtime(data_path)
        # 1) cached, fast
        rows = load_benefits_rows_cached(data_path, mtime)
        print(f"[INFO] Loaded {len(rows)} benefits rows from cache.")

        # Open async Postgres store & async checkpointer
        store_cm, store = await _open_store(pg_conn, index_cfg)
        ckpt_cm, checkpointer = await _open_checkpointer(ckpt_backend, pg_conn, sqlite_file)

        # Build agent (loads the official Excel within your builder)
        agent = build_agent(file_path=data_path, checkpointer=checkpointer, store=store, preloaded_rows=rows)

        # Save in session
        cl.user_session.set("agent", agent)
        cl.user_session.set("config_base", {
            "configurable": {
                "langgraph_user_id": langgraph_user_id,
                "thread_id": thread_id,
            }
        })
        cl.user_session.set("store_cm", store_cm)
        cl.user_session.set("ckpt_cm", ckpt_cm)

        await cl.Message(
            content="Hi!👋 I am your **Mazaya Benefits Agent**.\nAsk about offers by city/category, or share your preferences for personalized picks.",
            author="Saudia",
        ).send()

    except Exception as e:
        await cl.Message(content=f"Init error: {e}").send()
        traceback.print_exc()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
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
    # temp buffers for streamed tool args; key is tool_call_id if known, else index-based
    arg_buffers: dict[str, str] = {}

    def _key_for_chunk(tcc: dict, step_no: int) -> str:
        # Prefer id; else fall back to a stable per-step index key
        return tcc.get("id") or f"idx:{tcc.get('index', 0)}:step:{step_no}"

    async def _open_tool_step(tc_id: str, name: str, seed_args: str = ""):
        if tc_id in open_steps:
            return
        cm = cl.Step(name=f"🔧 {name or 'tool'}", type="tool")
        step = await cm.__aenter__()
        step.input = _pretty_json_text(seed_args)
        open_steps[tc_id] = (cm, step)

    async def _close_tool_step(tc_id: str, output_text: str):
        ctx = open_steps.pop(tc_id, None)
        if not ctx:
            # If we never opened (e.g., we didn't have id/name earlier), create & close now
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
            # Only need "messages" for this behavior
            async for msg_chunk, meta in agent.astream(
                {"messages": [{"role": "user", "content": message.content}]},
                config=config,
                stream_mode="messages",
            ):
                step_no = meta.get("langgraph_step", 0)

                # ---- 1) Assistant chunks (model output) ----
                if isinstance(msg_chunk, AIMessageChunk):
                    # (a) tool_calls (when id+name known) -> open/update steps
                    for tc in (getattr(msg_chunk, "tool_calls", []) or []):
                        # normalize fields across providers
                        name = tc.get("name") or tc.get("function", {}).get("name")
                        tc_id = tc.get("id")
                        idx   = tc.get("index", None)

                        if not (name and tc_id):
                            continue

                        # ---- NEW: migrate any earlier index-based buffer into the real id buffer
                        if idx is not None:
                            idx_key = f"idx:{idx}:step:{step_no}"
                            if idx_key in arg_buffers:
                                arg_buffers[tc_id] = arg_buffers.get(tc_id, "") + arg_buffers.pop(idx_key)

                        # open the step if needed and seed with whatever we have now
                        seed = arg_buffers.get(tc_id, "")
                        await _open_tool_step(tc_id, name, seed_args=seed)


                    # (b) tool_call_chunks -> stream JSON args pieces into the matching step
                    for tcc in (getattr(msg_chunk, "tool_call_chunks", []) or []):
                        key = _key_for_chunk(tcc, step_no)
                        arg_buffers[key] = arg_buffers.get(key, "") + (tcc.get("args") or "")
                        # if this chunk already has an id and we opened a step, mirror args there
                        tc_id = tcc.get("id")
                        if tc_id and tc_id in open_steps:
                            _, step = open_steps[tc_id]
                            step.input = arg_buffers[key]

                    # (c) natural language tokens -> stream to chat bubble
                    text = getattr(msg_chunk, "content", "")
                    if isinstance(text, str) and text:
                        await ai_msg.stream_token(text)
                    elif isinstance(text, list):
                        for part in text:
                            if isinstance(part, dict) and "text" in part and part["text"]:
                                await ai_msg.stream_token(part["text"])

                # ---- 2) Tool results (ToolMessage) ----
                elif isinstance(msg_chunk, ToolMessage) or getattr(msg_chunk, "type", None) == "tool":
                    tc_id = getattr(msg_chunk, "tool_call_id", None) or getattr(msg_chunk, "id", None)
                    out = getattr(msg_chunk, "content", "")
                    # If we only had an index key earlier, move its buffer under the real id now
                    # (arg buffers under "idx:...:step:..." are best-effort during early streaming)
                    if tc_id and tc_id not in arg_buffers:
                        # try to find any idx buffer from this step to attach
                        for k in list(arg_buffers.keys()):
                            if k.endswith(f":step:{meta.get('langgraph_step', 0)}"):
                                arg_buffers[tc_id] = arg_buffers.pop(k)
                                break
                    await _close_tool_step(tc_id or "unknown", out or "")

                # else: ignore other message types

        except Exception as e:
            await ai_msg.stream_token(f"\n\n[Generation error: {e}]")

        think.output = "Response generated."
        think.name="Thought"
        await think.update()

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
