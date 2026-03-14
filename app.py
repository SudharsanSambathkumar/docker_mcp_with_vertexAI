"""
Docker MCP — Streamlit UI
Powered by Gemini 2.5 Flash + MCP stdio

Run:
    streamlit run app.py

Requirements:
    pip install streamlit google-genai mcp anyio httpx
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator

import streamlit as st
from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_MODEL      = "gemini-2.5-flash"
SERVER_SCRIPT     = Path(__file__).parent / "server.py"
MAX_TOOL_ROUNDS   = 10
MAX_OUTPUT_TOKENS = 8192

SYSTEM_PROMPT = """You are DockerAI, an expert DevOps assistant that manages Docker infrastructure
through a set of MCP tools. You have full access to Docker: images, containers,
volumes, networks, builds, pushes, stats, logs, exec, compose and more.

Guidelines:
- Be concise but informative. Use markdown for structure.
- When a user asks to do something, do it immediately with the right tool(s).
- Chain multiple tool calls when needed (e.g. pull then run).
- Format sizes, timestamps, and port mappings in a human-friendly way.
- When showing container/image lists, use a clean markdown table.
- If a tool call fails, explain why and suggest a fix.
- Ask for confirmation only for destructive actions (remove, prune, system prune).
"""

# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class ToolEvent:
    tool_name: str
    args: dict
    result: str | None = None
    error: bool = False

@dataclass
class ChatMessage:
    role: str                        # "user" | "assistant" | "tool"
    content: str
    tool_events: list[ToolEvent] = field(default_factory=list)

# ─── Gemini schema helpers ────────────────────────────────────────────────────

def _json_type_to_gemini(pdef: dict) -> genai_types.Schema:
    jtype = pdef.get("type", "string")
    desc  = pdef.get("description", "")
    enum  = pdef.get("enum")

    if jtype == "object":
        sub = {k: _json_type_to_gemini(v) for k, v in pdef.get("properties", {}).items()}
        return genai_types.Schema(type=genai_types.Type.OBJECT, description=desc, properties=sub or None)
    if jtype == "array":
        return genai_types.Schema(type=genai_types.Type.ARRAY, description=desc,
                                   items=_json_type_to_gemini(pdef.get("items", {})))
    if jtype == "boolean":
        return genai_types.Schema(type=genai_types.Type.BOOLEAN, description=desc)
    if jtype == "integer":
        return genai_types.Schema(type=genai_types.Type.INTEGER, description=desc)
    if jtype == "number":
        return genai_types.Schema(type=genai_types.Type.NUMBER, description=desc)
    if enum:
        return genai_types.Schema(type=genai_types.Type.STRING, description=desc, enum=enum)
    return genai_types.Schema(type=genai_types.Type.STRING, description=desc)


def mcp_tool_to_gemini(tool) -> genai_types.FunctionDeclaration:
    schema = tool.inputSchema or {}
    props  = {k: _json_type_to_gemini(v) for k, v in schema.get("properties", {}).items()}
    params = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties=props,
        required=schema.get("required", []),
    ) if props else None
    return genai_types.FunctionDeclaration(
        name=tool.name,
        description=tool.description or "",
        parameters=params,
    )

# ─── Core async agent (runs inside a dedicated event loop thread) ─────────────

async def _run_agent_turn(
    api_key: str,
    user_message: str,
    gemini_history: list[genai_types.Content],
    on_event,                          # callback(event_type, payload)
) -> list[genai_types.Content]:
    """
    Full agentic turn: Gemini ↔ MCP tool calls.
    Emits events via on_event for the UI to consume.
    """
    gemini = genai.Client(api_key=api_key)

    server_params = StdioServerParameters(
        command="python",
        args=[str(SERVER_SCRIPT)],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools_resp = await session.list_tools()
            mcp_tools = mcp_tools_resp.tools

            fn_decls = [mcp_tool_to_gemini(t) for t in mcp_tools]
            gemini_tools = [genai_types.Tool(function_declarations=fn_decls)]

            # Append user message to history
            gemini_history.append(
                genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
            )

            for _ in range(MAX_TOOL_ROUNDS):
                on_event("thinking", None)

                response = gemini.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=gemini_history,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=gemini_tools,
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        temperature=0.2,
                    ),
                )

                candidate = response.candidates[0]
                parts = candidate.content.parts
                gemini_history.append(candidate.content)

                text_parts = [p.text for p in parts if p.text]
                fn_calls   = [p.function_call for p in parts if p.function_call]

                if text_parts:
                    on_event("text", "".join(text_parts))

                if not fn_calls:
                    break

                tool_resp_parts: list[genai_types.Part] = []

                for fn_call in fn_calls:
                    name = fn_call.name
                    args = dict(fn_call.args) if fn_call.args else {}
                    on_event("tool_call", {"name": name, "args": args})

                    try:
                        mcp_result = await session.call_tool(name, args)
                        result_text = "\n".join(
                            b.text for b in mcp_result.content if hasattr(b, "text")
                        ) or "(no output)"
                        on_event("tool_result", {"name": name, "result": result_text, "error": False})
                    except Exception as exc:
                        result_text = f"ERROR: {exc}"
                        on_event("tool_result", {"name": name, "result": result_text, "error": True})

                    tool_resp_parts.append(
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=name,
                                response={"result": result_text},
                            )
                        )
                    )

                gemini_history.append(
                    genai_types.Content(role="user", parts=tool_resp_parts)
                )

    return gemini_history


def run_agent_sync(api_key, user_message, gemini_history, event_queue) -> list:
    """Run the async agent in a fresh event loop (called from a thread)."""
    events_collected = []

    def on_event(kind, payload):
        events_collected.append((kind, payload))
        event_queue.put((kind, payload))

    loop = asyncio.new_event_loop()
    try:
        new_history = loop.run_until_complete(
            _run_agent_turn(api_key, user_message, gemini_history, on_event)
        )
    finally:
        loop.close()

    event_queue.put(("done", None))
    return new_history


# ─── Helpers to fetch tools list for sidebar ─────────────────────────────────

async def _fetch_tools():
    server_params = StdioServerParameters(command="python", args=[str(SERVER_SCRIPT)])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [(t.name, t.description or "") for t in resp.tools]


def fetch_tools_sync():
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_fetch_tools())
    finally:
        loop.close()

# ─── UI helpers ───────────────────────────────────────────────────────────────

def render_tool_event(ev: ToolEvent):
    if ev.result is None:
        # Still pending
        with st.expander(f"🔧 `{ev.tool_name}`", expanded=False):
            st.json(ev.args)
        return

    icon  = "✅" if not ev.error else "❌"
    label = f"{icon} `{ev.tool_name}`"
    with st.expander(label, expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("**Arguments**")
            st.json(ev.args)
        with col2:
            st.caption("**Result**")
            try:
                parsed = json.loads(ev.result)
                st.json(parsed)
            except (json.JSONDecodeError, TypeError):
                st.code(ev.result[:4000], language="text")


def render_message(msg: ChatMessage):
    if msg.role == "user":
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif msg.role == "assistant":
        with st.chat_message("assistant", avatar="🐳"):
            for ev in msg.tool_events:
                render_tool_event(ev)
            if msg.content:
                st.markdown(msg.content)


# ─── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Docker MCP",
    page_icon="🐳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] *        { color: #c9d1d9 !important; }

/* ── header ── */
.dock-header {
    display: flex; align-items: center; gap: 12px;
    padding: 16px 0 8px 0; border-bottom: 1px solid #21262d; margin-bottom: 20px;
}
.dock-header h1 { margin: 0; font-size: 1.6rem; color: #58a6ff; }
.dock-badge {
    background: #1f6feb; color: #fff; font-size: 0.7rem;
    padding: 2px 8px; border-radius: 12px; font-weight: 600; letter-spacing: .05em;
}

/* ── chat messages ── */
[data-testid="stChatMessage"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
}

/* ── expander (tool events) ── */
[data-testid="stExpander"] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    margin-bottom: 6px !important;
}
[data-testid="stExpander"] summary { color: #8b949e !important; font-size: 0.85rem; }

/* ── chat input ── */
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea:focus { border-color: #58a6ff !important; }

/* ── code blocks ── */
pre, code { background: #0d1117 !important; }

/* ── sidebar tool list ── */
.tool-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 5px;
    font-size: 0.78rem;
}
.tool-name { color: #79c0ff; font-weight: 600; font-family: monospace; }
.tool-desc { color: #8b949e; margin-top: 2px; }

/* ── status badge ── */
.status-connected { color: #3fb950; font-size: 0.8rem; font-weight: 600; }
.status-error     { color: #f85149; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─── Session state init ───────────────────────────────────────────────────────

if "messages"        not in st.session_state: st.session_state.messages        = []
if "gemini_history"  not in st.session_state: st.session_state.gemini_history  = []
if "api_key"         not in st.session_state: st.session_state.api_key         = os.environ.get("GEMINI_API_KEY", "")
if "tools_cache"     not in st.session_state: st.session_state.tools_cache     = None
if "processing"      not in st.session_state: st.session_state.processing      = False

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🐳 Docker MCP")
    st.markdown("---")

    # API key input
    api_key_input = st.text_input(
        "Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIza...",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    st.markdown("---")

    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    quick_prompts = [
        ("📋 List images",        "List all local Docker images with their sizes and tags."),
        ("🐋 List containers",    "Show all running containers with their status and ports."),
        ("💾 List volumes",       "List all Docker volumes."),
        ("🌐 List networks",      "List all Docker networks."),
        ("📊 System info",        "Show Docker system info and resource usage."),
        ("🧹 Disk usage",         "Show Docker disk usage breakdown."),
    ]
    for label, prompt in quick_prompts:
        if st.button(label, use_container_width=True, disabled=st.session_state.processing):
            st.session_state["pending_prompt"] = prompt

    st.markdown("---")

    # Tools list
    st.markdown("### 🔧 Available Tools")

    if st.button("🔄 Refresh tools", use_container_width=True):
        st.session_state.tools_cache = None

    if st.session_state.tools_cache is None:
        with st.spinner("Loading tools…"):
            try:
                st.session_state.tools_cache = fetch_tools_sync()
                st.markdown('<p class="status-connected">● MCP server connected</p>', unsafe_allow_html=True)
            except Exception as e:
                st.session_state.tools_cache = []
                st.markdown(f'<p class="status-error">● MCP error: {str(e)[:60]}</p>', unsafe_allow_html=True)

    if st.session_state.tools_cache:
        # Group tools by category
        categories = {
            "Images":     [t for t in st.session_state.tools_cache if "image" in t[0]],
            "Containers": [t for t in st.session_state.tools_cache if "container" in t[0]],
            "Networks":   [t for t in st.session_state.tools_cache if "network" in t[0]],
            "Volumes":    [t for t in st.session_state.tools_cache if "volume" in t[0]],
            "System":     [t for t in st.session_state.tools_cache if "system" in t[0] or "compose" in t[0]],
        }
        for cat, tools in categories.items():
            if tools:
                with st.expander(f"**{cat}** ({len(tools)})", expanded=False):
                    for name, desc in tools:
                        st.markdown(
                            f'<div class="tool-item">'
                            f'<div class="tool-name">{name}</div>'
                            f'<div class="tool-desc">{desc[:70]}{"…" if len(desc)>70 else ""}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    st.markdown("---")

    # Clear chat
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.gemini_history = []
        st.rerun()

    st.markdown(
        '<div style="color:#8b949e;font-size:0.72rem;margin-top:8px">'
        'gemini-2.5-flash · MCP stdio · docker-py 7.x'
        '</div>',
        unsafe_allow_html=True,
    )


# ─── Main area ────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="dock-header">'
    '<span style="font-size:2rem">🐳</span>'
    '<h1>Docker MCP</h1>'
    '<span class="dock-badge">Gemini 2.5 Flash</span>'
    '</div>',
    unsafe_allow_html=True,
)

# Render existing conversation
for msg in st.session_state.messages:
    render_message(msg)

# ── Handle pending quick-action prompt ───────────────────────────────────────
pending = st.session_state.pop("pending_prompt", None)

# ── Chat input ────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask anything about your Docker environment…",
    disabled=st.session_state.processing,
)

prompt = pending or user_input

if prompt and st.session_state.api_key and not st.session_state.processing:
    # Add user message to display
    user_msg = ChatMessage(role="user", content=prompt)
    st.session_state.messages.append(user_msg)

    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for assistant response
    with st.chat_message("assistant", avatar="🐳"):
        status_placeholder = st.empty()
        tool_placeholder   = st.empty()
        text_placeholder   = st.empty()

        status_placeholder.markdown("*🤔 Thinking…*")

        # Collect events from the agent thread
        import queue as _queue
        event_queue = _queue.Queue()

        # Build a mutable copy of history for the thread
        history_copy = list(st.session_state.gemini_history)

        agent_result: dict = {}

        def _thread_target():
            new_hist = run_agent_sync(
                api_key=st.session_state.api_key,
                user_message=prompt,
                gemini_history=history_copy,
                event_queue=event_queue,
            )
            agent_result["history"] = new_hist

        thread = threading.Thread(target=_thread_target, daemon=True)
        thread.start()

        # Drain event queue, updating UI
        assistant_msg = ChatMessage(role="assistant", content="")
        current_tool_ev: ToolEvent | None = None
        final_text = ""

        while True:
            try:
                kind, payload = event_queue.get(timeout=0.1)
            except _queue.Empty:
                # Keep spinner alive
                continue

            if kind == "done":
                break

            elif kind == "thinking":
                status_placeholder.markdown("*🤔 Gemini is thinking…*")

            elif kind == "text":
                final_text = payload
                status_placeholder.empty()
                text_placeholder.markdown(payload)

            elif kind == "tool_call":
                status_placeholder.markdown(f"*🔧 Calling `{payload['name']}`…*")
                ev = ToolEvent(tool_name=payload["name"], args=payload["args"])
                assistant_msg.tool_events.append(ev)
                current_tool_ev = ev
                # Re-render all tool events so far
                with tool_placeholder.container():
                    for e in assistant_msg.tool_events:
                        render_tool_event(e)

            elif kind == "tool_result":
                if current_tool_ev and current_tool_ev.tool_name == payload["name"]:
                    current_tool_ev.result = payload["result"]
                    current_tool_ev.error  = payload["error"]
                status_placeholder.markdown("*🤔 Processing result…*")
                with tool_placeholder.container():
                    for e in assistant_msg.tool_events:
                        render_tool_event(e)

        thread.join()
        status_placeholder.empty()

        # Finalise assistant message
        assistant_msg.content = final_text
        st.session_state.messages.append(assistant_msg)
        st.session_state.gemini_history = agent_result.get("history", history_copy)

        # Final render
        with tool_placeholder.container():
            for ev in assistant_msg.tool_events:
                render_tool_event(ev)
        text_placeholder.markdown(final_text)

elif prompt and not st.session_state.api_key:
    st.warning("⚠️ Please enter your **Gemini API Key** in the sidebar first.")

# ── Empty state hint ─────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div style="text-align:center;padding:60px 20px;color:#8b949e">
    <div style="font-size:4rem;margin-bottom:16px">🐳</div>
    <h3 style="color:#c9d1d9;margin-bottom:8px">Docker MCP Assistant</h3>
    <p style="max-width:480px;margin:0 auto;line-height:1.6">
        Ask anything in plain English — pull images, run containers, tail logs,
        inspect stats, manage volumes & networks, build from a Dockerfile, and more.
    </p>
    <div style="margin-top:28px;display:flex;flex-wrap:wrap;gap:10px;justify-content:center">
        <code style="background:#161b22;border:1px solid #30363d;padding:6px 12px;border-radius:6px;font-size:0.82rem">
            Pull nginx:alpine and run it on port 8080
        </code>
        <code style="background:#161b22;border:1px solid #30363d;padding:6px 12px;border-radius:6px;font-size:0.82rem">
            Show me stats for all running containers
        </code>
        <code style="background:#161b22;border:1px solid #30363d;padding:6px 12px;border-radius:6px;font-size:0.82rem">
            Build an image from this Dockerfile: …
        </code>
        <code style="background:#161b22;border:1px solid #30363d;padding:6px 12px;border-radius:6px;font-size:0.82rem">
            Prune all stopped containers and dangling images
        </code>
    </div>
</div>
""", unsafe_allow_html=True)
