 """
Docker MCP — Streamlit UI
Powered by Gemini 2.5 Flash on Vertex AI (ADC / Cloud Run — no API key needed)

Run locally:
    gcloud auth application-default login
    streamlit run app.py

Deploy to Cloud Run:
    gcloud run deploy docker-mcp-ui --source . --region us-central1

Environment variables (all optional):
    GCP_PROJECT   — GCP project ID (auto-detected on Cloud Run)
    GCP_LOCATION  — Vertex AI region  (default: us-central1)
    GEMINI_MODEL  — override model    (default: gemini-2.5-flash)

Requirements:
    pip install streamlit "google-genai[vertexai]" mcp anyio httpx
"""

from __future__ import annotations

import asyncio
import json
import os
import queue as _queue
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types as genai_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_MODEL  = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GCP_PROJECT   = os.environ.get("GCP_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT", ""))
GCP_LOCATION  = os.environ.get("GCP_LOCATION", "us-central1")
SERVER_SCRIPT = Path(__file__).parent / "server.py"
MAX_ROUNDS    = 10
MAX_TOKENS    = 8192

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
    role: str          # "user" | "assistant"
    content: str
    tool_events: list[ToolEvent] = field(default_factory=list)

# ─── Vertex AI client (ADC — no API key) ─────────────────────────────────────

def _detect_project() -> str:
    """Try to get GCP project from metadata server (Cloud Run / GCE)."""
    try:
        req = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
        )
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read().decode()
    except Exception:
        return ""


def build_vertex_client() -> genai.Client:
    project = GCP_PROJECT or _detect_project()
    return genai.Client(
        vertexai=True,
        project=project or None,
        location=GCP_LOCATION,
    )

# ─── Gemini schema conversion ─────────────────────────────────────────────────

def _json_to_gemini(pdef: dict) -> genai_types.Schema:
    jtype, desc, enum = pdef.get("type", "string"), pdef.get("description", ""), pdef.get("enum")
    if jtype == "object":
        sub = {k: _json_to_gemini(v) for k, v in pdef.get("properties", {}).items()}
        return genai_types.Schema(type=genai_types.Type.OBJECT, description=desc, properties=sub or None)
    if jtype == "array":
        return genai_types.Schema(type=genai_types.Type.ARRAY, description=desc,
                                  items=_json_to_gemini(pdef.get("items", {})))
    type_map = {"boolean": genai_types.Type.BOOLEAN,
                "integer": genai_types.Type.INTEGER,
                "number":  genai_types.Type.NUMBER}
    if jtype in type_map:
        return genai_types.Schema(type=type_map[jtype], description=desc)
    return genai_types.Schema(type=genai_types.Type.STRING, description=desc, enum=enum)


def mcp_to_gemini(tool) -> genai_types.FunctionDeclaration:
    schema = tool.inputSchema or {}
    props  = {k: _json_to_gemini(v) for k, v in schema.get("properties", {}).items()}
    params = genai_types.Schema(type=genai_types.Type.OBJECT, properties=props,
                                required=schema.get("required", [])) if props else None
    return genai_types.FunctionDeclaration(name=tool.name,
                                           description=tool.description or "",
                                           parameters=params)

# ─── Async agent ──────────────────────────────────────────────────────────────

async def _agent_turn(user_message: str, history: list, on_event) -> list:
    client = build_vertex_client()
    async with stdio_client(StdioServerParameters(command="python",
                                                   args=[str(SERVER_SCRIPT)])) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            tools     = (await session.list_tools()).tools
            g_tools   = [genai_types.Tool(function_declarations=[mcp_to_gemini(t) for t in tools])]
            history.append(genai_types.Content(role="user",
                                               parts=[genai_types.Part(text=user_message)]))

            for _ in range(MAX_ROUNDS):
                on_event("thinking", None)
                resp      = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=history,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=g_tools,
                        max_output_tokens=MAX_TOKENS,
                        temperature=0.2,
                    ),
                )
                candidate = resp.candidates[0]
                parts     = candidate.content.parts
                history.append(candidate.content)

                texts    = [p.text for p in parts if p.text]
                fn_calls = [p.function_call for p in parts if p.function_call]

                if texts:
                    on_event("text", "".join(texts))
                if not fn_calls:
                    break

                resp_parts: list[genai_types.Part] = []
                for fn in fn_calls:
                    name, args = fn.name, dict(fn.args) if fn.args else {}
                    on_event("tool_call", {"name": name, "args": args})
                    try:
                        res         = await session.call_tool(name, args)
                        result_text = "\n".join(b.text for b in res.content
                                                if hasattr(b, "text")) or "(no output)"
                        on_event("tool_result", {"name": name, "result": result_text, "error": False})
                    except Exception as exc:
                        result_text = f"ERROR: {exc}"
                        on_event("tool_result", {"name": name, "result": result_text, "error": True})
                    resp_parts.append(genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=name, response={"result": result_text})))
                history.append(genai_types.Content(role="user", parts=resp_parts))
    return history


def run_agent_sync(msg: str, history: list, eq: _queue.Queue) -> list:
    def cb(kind, payload): eq.put((kind, payload))
    loop = asyncio.new_event_loop()
    box: dict = {}
    try:
        box["h"] = loop.run_until_complete(_agent_turn(msg, history, cb))
    except Exception as exc:
        eq.put(("error", str(exc)))
        box["h"] = history
    finally:
        loop.close()
    eq.put(("done", None))
    return box.get("h", history)


async def _fetch_tools():
    async with stdio_client(StdioServerParameters(command="python",
                                                   args=[str(SERVER_SCRIPT)])) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            return [(t.name, t.description or "") for t in (await session.list_tools()).tools]

def fetch_tools_sync():
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_fetch_tools())
    finally:
        loop.close()

# ─── Streamlit page ───────────────────────────────────────────────────────────

st.set_page_config(page_title="Docker MCP", page_icon="🐳",
                   layout="wide", initial_sidebar_state="expanded")

# ── Light theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
/* base */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container { background:#f8fafc !important; color:#1e293b !important; }

/* sidebar */
[data-testid="stSidebar"] {
    background:#ffffff !important;
    border-right:1px solid #e2e8f0 !important;
}
[data-testid="stSidebar"] * { color:#334155 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:#0f172a !important; font-weight:700; }

/* sidebar buttons */
[data-testid="stSidebar"] .stButton button {
    background:#f1f5f9 !important; border:1px solid #e2e8f0 !important;
    color:#334155 !important; border-radius:8px !important;
    font-size:.84rem !important; transition:all .15s;
}
[data-testid="stSidebar"] .stButton button:hover {
    background:#e0f2fe !important; border-color:#38bdf8 !important; color:#0369a1 !important;
}

/* header */
.dock-header {
    display:flex; align-items:center; gap:14px;
    padding:20px 0 12px; border-bottom:2px solid #e2e8f0; margin-bottom:24px;
}
.dock-header h1 { margin:0; font-size:1.7rem; color:#0284c7; font-weight:800; }
.dock-badge {
    background:linear-gradient(135deg,#0284c7,#0ea5e9);
    color:#fff; font-size:.7rem; padding:3px 10px; border-radius:20px;
    font-weight:700; letter-spacing:.06em;
    box-shadow:0 1px 4px rgba(2,132,199,.3);
}

/* chat bubbles */
[data-testid="stChatMessage"] {
    background:#ffffff !important; border:1px solid #e2e8f0 !important;
    border-radius:12px !important; margin-bottom:10px !important;
    box-shadow:0 1px 3px rgba(0,0,0,.05) !important;
}

/* tool expanders */
[data-testid="stExpander"] {
    background:#f0f9ff !important; border:1px solid #bae6fd !important;
    border-radius:8px !important; margin-bottom:6px !important;
}
[data-testid="stExpander"] summary {
    color:#0369a1 !important; font-size:.85rem !important; font-weight:600 !important;
}

/* chat input */
[data-testid="stChatInput"] textarea {
    background:#ffffff !important; border:1.5px solid #cbd5e1 !important;
    border-radius:12px !important; color:#1e293b !important; font-size:.95rem !important;
    box-shadow:0 1px 4px rgba(0,0,0,.06) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color:#0284c7 !important; box-shadow:0 0 0 3px rgba(2,132,199,.15) !important;
}

/* text inputs */
.stTextInput input {
    background:#fff !important; border:1.5px solid #cbd5e1 !important;
    border-radius:8px !important; color:#1e293b !important;
}
.stTextInput input:focus { border-color:#0284c7 !important; }

/* code */
pre, code {
    background:#f1f5f9 !important; color:#0f172a !important;
    border:1px solid #e2e8f0 !important;
}

/* tool cards */
.tool-card {
    background:#f8fafc; border:1px solid #e2e8f0; border-left:3px solid #0284c7;
    border-radius:6px; padding:7px 11px; margin-bottom:5px; font-size:.78rem;
}
.tool-name { color:#0284c7; font-weight:700; font-family:monospace; }
.tool-desc { color:#64748b; margin-top:2px; line-height:1.4; }

/* status */
.s-ok  { color:#16a34a; font-size:.8rem; font-weight:600; }
.s-err { color:#dc2626; font-size:.8rem; font-weight:600; }

/* clear btn */
.clear-btn button {
    background:#fff1f2 !important; border:1px solid #fca5a5 !important;
    color:#dc2626 !important; border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Session state ────────────────────────────────────────────────────────────
for k, v in [("messages", []), ("gemini_history", []),
             ("tools_cache", None), ("processing", False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🐳 Docker MCP")

    # ── Vertex AI status ──────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="background:#f0fdf4;border:1px solid #bbf7d0;
                        border-radius:8px;padding:10px 14px;margin-bottom:8px">
            <div class="s-ok">● Vertex AI — ADC</div>
            <div style="color:#64748b;font-size:.75rem;margin-top:3px">
                No API key · Application Default Credentials
            </div>
            <div style="color:#94a3b8;font-size:.72rem;margin-top:2px">
                Project: <b>{GCP_PROJECT or "auto-detect"}</b> &nbsp;·&nbsp;
                Region: <b>{GCP_LOCATION}</b>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Docker host status ────────────────────────────────────────────────────
    _docker_host = os.environ.get("DOCKER_HOST", "")
    if _docker_host:
        st.markdown(
            f"""<div style="background:#f0fdf4;border:1px solid #bbf7d0;
                            border-radius:8px;padding:10px 14px;margin-bottom:8px">
                <div class="s-ok">● Docker Daemon — TCP</div>
                <div style="color:#94a3b8;font-size:.72rem;margin-top:3px;
                            word-break:break-all;font-family:monospace">
                    {_docker_host}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="background:#fff7ed;border:1px solid #fed7aa;
                           border-radius:8px;padding:10px 14px;margin-bottom:8px">
                <div style="color:#c2410c;font-size:.8rem;font-weight:600">
                    ⚠ DOCKER_HOST not set
                </div>
                <div style="color:#64748b;font-size:.74rem;margin-top:5px;line-height:1.6">
                    Set this env var in Cloud Run:<br>
                    <code style="background:#fef3c7;padding:2px 5px;border-radius:4px;
                                 font-size:.7rem;color:#92400e">
                        DOCKER_HOST=tcp://YOUR_VM_IP:2375
                    </code><br>
                    Run <b>docker-daemon-setup.sh</b> on your VM first.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    for label, prompt in [
        ("📋 List images",     "List all local Docker images with their sizes and tags."),
        ("🐋 List containers", "Show all running containers with their status and ports."),
        ("💾 List volumes",    "List all Docker volumes."),
        ("🌐 List networks",   "List all Docker networks."),
        ("📊 System info",     "Show Docker system info and resource usage."),
        ("🧹 Disk usage",      "Show Docker disk usage breakdown."),
    ]:
        if st.button(label, use_container_width=True, disabled=st.session_state.processing):
            st.session_state["pending_prompt"] = prompt

    st.markdown("---")
    st.markdown("### 🔧 Tools")
    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("↺", help="Refresh"):
            st.session_state.tools_cache = None

    if st.session_state.tools_cache is None:
        with st.spinner("Connecting…"):
            try:
                st.session_state.tools_cache = fetch_tools_sync()
                with c1:
                    st.markdown('<span class="s-ok">● Connected</span>', unsafe_allow_html=True)
            except Exception as e:
                st.session_state.tools_cache = []
                with c1:
                    st.markdown(f'<span class="s-err">● {str(e)[:45]}</span>', unsafe_allow_html=True)

    cats = {
        "📦 Images":     [t for t in (st.session_state.tools_cache or []) if "image"     in t[0]],
        "🐋 Containers": [t for t in (st.session_state.tools_cache or []) if "container" in t[0]],
        "🌐 Networks":   [t for t in (st.session_state.tools_cache or []) if "network"   in t[0]],
        "💾 Volumes":    [t for t in (st.session_state.tools_cache or []) if "volume"    in t[0]],
        "⚙️ System":     [t for t in (st.session_state.tools_cache or [])
                          if "system" in t[0] or "compose" in t[0]],
    }
    for cat, tools in cats.items():
        if not tools:
            continue
        with st.expander(f"{cat} ({len(tools)})", expanded=False):
            for name, desc in tools:
                st.markdown(
                    f'<div class="tool-card"><div class="tool-name">{name}</div>'
                    f'<div class="tool-desc">{desc[:72]}{"…" if len(desc)>72 else ""}</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.gemini_history = []
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#94a3b8;font-size:.72rem;margin-top:10px">'
        f'{GEMINI_MODEL} · Vertex AI · MCP stdio</div>',
        unsafe_allow_html=True,
    )

# ─── Beautiful result renderers ───────────────────────────────────────────────

def _status_badge(status: str) -> str:
    colors = {
        "running":  ("#dcfce7", "#16a34a", "▶"),
        "exited":   ("#fee2e2", "#dc2626", "■"),
        "paused":   ("#fef9c3", "#ca8a04", "⏸"),
        "created":  ("#e0f2fe", "#0284c7", "○"),
        "dead":     ("#fecaca", "#991b1b", "✕"),
    }
    bg, fg, icon = colors.get(status.lower(), ("#f1f5f9", "#64748b", "?"))
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:20px;'
            f'font-size:.72rem;font-weight:700">{icon} {status}</span>')


def _tag_chip(tag: str, bg="#e0f2fe", fg="#0369a1") -> str:
    return (f'<span style="background:{bg};color:{fg};padding:2px 8px;border-radius:4px;'
            f'font-size:.72rem;font-family:monospace;margin-right:4px">{tag}</span>')


def _render_image_list(data: list) -> None:
    st.markdown(f'<div style="color:#64748b;font-size:.8rem;margin-bottom:10px">'
                f'<b>{len(data)}</b> images found</div>', unsafe_allow_html=True)
    for img in data:
        tags = img.get("tags") or ["<none>"]
        primary = tags[0]
        extra   = tags[1:]
        created = img.get("created", "")[:10]
        arch    = img.get("architecture", "")
        size    = img.get("size", "")
        img_id  = img.get("id", "")[:14]

        tag_html = "".join(_tag_chip(t) for t in tags)
        extra_html = ""
        if extra:
            extra_html = "".join(_tag_chip(t, "#f1f5f9", "#64748b") for t in extra)

        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:12px 16px;margin-bottom:8px;
                    box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div>
                    <div style="font-weight:700;color:#0f172a;font-size:.95rem;
                                font-family:monospace">{primary}</div>
                    <div style="margin-top:5px">{tag_html}{extra_html}</div>
                </div>
                <div style="text-align:right;flex-shrink:0;margin-left:16px">
                    <div style="font-size:1rem;font-weight:700;color:#0284c7">{size}</div>
                    <div style="color:#94a3b8;font-size:.72rem;margin-top:2px">{created}</div>
                </div>
            </div>
            <div style="margin-top:8px;color:#94a3b8;font-size:.72rem">
                ID: <code style="background:#f1f5f9;padding:1px 5px;border-radius:3px">{img_id}</code>
                &nbsp;·&nbsp; {arch}
            </div>
        </div>""", unsafe_allow_html=True)


def _render_container_list(data: list) -> None:
    st.markdown(f'<div style="color:#64748b;font-size:.8rem;margin-bottom:10px">'
                f'<b>{len(data)}</b> containers</div>', unsafe_allow_html=True)
    for c in data:
        name    = c.get("name", "?")
        status  = c.get("status", "unknown")
        image   = c.get("image", "")
        cid     = c.get("id", "")[:12]
        created = c.get("created", "")[:10]
        ports   = c.get("ports", {})
        nets    = c.get("networks", [])

        port_html = ""
        for proto, host_ports in ports.items():
            for hp in (host_ports or []):
                port_html += _tag_chip(f"{hp}→{proto}", "#f0fdf4", "#16a34a")

        net_html = "".join(_tag_chip(n, "#faf5ff", "#7c3aed") for n in nets)

        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:12px 16px;margin-bottom:8px;
                    box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="font-weight:700;color:#0f172a;font-size:.95rem">{name}</div>
                <div>{_status_badge(status)}</div>
            </div>
            <div style="color:#64748b;font-size:.8rem;margin-top:4px;font-family:monospace">{image}</div>
            <div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:4px">
                {port_html}{net_html}
            </div>
            <div style="margin-top:6px;color:#94a3b8;font-size:.72rem">
                ID: <code style="background:#f1f5f9;padding:1px 5px;border-radius:3px">{cid}</code>
                &nbsp;·&nbsp; created {created}
            </div>
        </div>""", unsafe_allow_html=True)


def _render_volume_list(data: list) -> None:
    st.markdown(f'<div style="color:#64748b;font-size:.8rem;margin-bottom:10px">'
                f'<b>{len(data)}</b> volumes</div>', unsafe_allow_html=True)
    for v in data:
        name   = v.get("name", "?")
        driver = v.get("driver", "local")
        mount  = v.get("mountpoint", "")
        created = v.get("created", "")[:10] if v.get("created") else ""
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:12px 16px;margin-bottom:6px;
                    box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="font-weight:700;color:#0f172a;font-family:monospace;font-size:.88rem">{name}</div>
                {_tag_chip(driver, "#faf5ff", "#7c3aed")}
            </div>
            <div style="color:#94a3b8;font-size:.72rem;margin-top:6px;font-family:monospace;
                        word-break:break-all">{mount}</div>
            {"<div style='color:#94a3b8;font-size:.72rem;margin-top:2px'>created " + created + "</div>" if created else ""}
        </div>""", unsafe_allow_html=True)


def _render_network_list(data: list) -> None:
    st.markdown(f'<div style="color:#64748b;font-size:.8rem;margin-bottom:10px">'
                f'<b>{len(data)}</b> networks</div>', unsafe_allow_html=True)
    for n in data:
        name     = n.get("name", "?")
        driver   = n.get("driver", "")
        scope    = n.get("scope", "")
        internal = n.get("internal", False)
        conts    = n.get("containers", [])
        badge    = _tag_chip(driver, "#e0f2fe", "#0369a1")
        scope_b  = _tag_chip(scope, "#f1f5f9", "#64748b")
        int_b    = _tag_chip("internal", "#fff7ed", "#c2410c") if internal else ""
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                    padding:12px 16px;margin-bottom:6px;
                    box-shadow:0 1px 3px rgba(0,0,0,.04)">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="font-weight:700;color:#0f172a;font-size:.9rem">{name}</div>
                <div>{badge}{scope_b}{int_b}</div>
            </div>
            {"<div style='color:#94a3b8;font-size:.72rem;margin-top:6px'>" + str(len(conts)) + " container(s) attached</div>" if conts else ""}
        </div>""", unsafe_allow_html=True)


def _render_stats(data: dict) -> None:
    metrics = [
        ("🖥️ CPU",    data.get("cpu_percent", 0),    f"{data.get('cpu_percent', 0)}%",         100,  "#0284c7"),
        ("💾 Memory", data.get("memory_percent", 0), data.get("memory_usage", ""),              100,  "#7c3aed"),
    ]
    st.markdown(f'<div style="font-weight:700;color:#0f172a;margin-bottom:10px">'
                f'📊 {data.get("container", "")}</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, (label, pct, val, max_val, color) in enumerate(metrics):
        bar_w = min(int(float(pct)), 100)
        with cols[i]:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
                        padding:14px;box-shadow:0 1px 3px rgba(0,0,0,.04)">
                <div style="color:#64748b;font-size:.8rem">{label}</div>
                <div style="font-size:1.4rem;font-weight:800;color:{color};margin:4px 0">{val}</div>
                <div style="background:#f1f5f9;border-radius:4px;height:6px;margin-top:6px">
                    <div style="background:{color};width:{bar_w}%;height:6px;border-radius:4px"></div>
                </div>
                <div style="color:#94a3b8;font-size:.7rem;margin-top:3px">{pct}%</div>
            </div>""", unsafe_allow_html=True)

    # Network + Block IO row
    net_cols = st.columns(4)
    for col, (label, val) in zip(net_cols, [
        ("⬇ Net RX",    data.get("network_rx", "0 B")),
        ("⬆ Net TX",    data.get("network_tx", "0 B")),
        ("📖 Disk Read", data.get("block_read", "0 B")),
        ("✏ Disk Write", data.get("block_write", "0 B")),
    ]):
        with col:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
                        padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,.04)">
                <div style="color:#94a3b8;font-size:.72rem">{label}</div>
                <div style="font-size:.95rem;font-weight:700;color:#0f172a;margin-top:2px">{val}</div>
            </div>""", unsafe_allow_html=True)


def _render_system_info(data: dict) -> None:
    cards = [
        ("🐳 Docker",     data.get("docker_version", ""), "version"),
        ("🖥️ OS",         data.get("operating_system", ""), ""),
        ("⚙️ Arch",       data.get("architecture", ""), ""),
        ("🧠 Memory",     data.get("memory", ""), ""),
        ("💻 CPUs",       str(data.get("cpus", "")), ""),
        ("📦 Images",     str(data.get("images", "")), ""),
        ("🐋 Containers", str(data.get("containers", "")), ""),
        ("▶ Running",    str(data.get("containers_running", "")), ""),
        ("💽 Storage",    data.get("storage_driver", ""), "driver"),
        ("🐧 Kernel",     data.get("kernel_version", ""), ""),
    ]
    cols = st.columns(3)
    for i, (label, val, hint) in enumerate(cards):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
                        padding:12px;margin-bottom:8px;box-shadow:0 1px 3px rgba(0,0,0,.04)">
                <div style="color:#94a3b8;font-size:.72rem">{label}</div>
                <div style="font-size:.92rem;font-weight:700;color:#0f172a;margin-top:3px;
                            word-break:break-all">{val or "—"}</div>
            </div>""", unsafe_allow_html=True)


def _render_inspect(data: dict) -> None:
    """Generic inspect view — key/value grid."""
    skip = {"full_id", "labels"}
    items = [(k, v) for k, v in data.items() if k not in skip and v not in (None, "", [], {})]
    cols = st.columns(2)
    for i, (k, v) in enumerate(items):
        val = json.dumps(v) if isinstance(v, (list, dict)) else str(v)
        with cols[i % 2]:
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
                        padding:10px 12px;margin-bottom:6px">
                <div style="color:#94a3b8;font-size:.72rem;text-transform:uppercase;
                            letter-spacing:.05em">{k.replace("_", " ")}</div>
                <div style="font-size:.85rem;font-weight:600;color:#0f172a;margin-top:3px;
                            word-break:break-all;font-family:monospace">{val[:120]}</div>
            </div>""", unsafe_allow_html=True)


def _render_history(rows: list) -> None:
    st.markdown(f'<div style="color:#64748b;font-size:.8rem;margin-bottom:8px">'
                f'<b>{len(rows)}</b> layers</div>', unsafe_allow_html=True)
    for row in rows:
        size    = row.get("size", "")
        cmd     = row.get("created_by", "")
        created = row.get("created", "")[:10]
        lid     = row.get("id", "")[:12]
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
                    padding:9px 13px;margin-bottom:4px;font-size:.78rem">
            <div style="display:flex;justify-content:space-between">
                <code style="color:#0369a1;background:#e0f2fe;padding:1px 6px;
                             border-radius:3px">{lid or "<missing>"}</code>
                <span style="color:#0284c7;font-weight:700">{size}</span>
            </div>
            <div style="color:#475569;margin-top:5px;font-family:monospace;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{cmd}</div>
            <div style="color:#94a3b8;margin-top:3px">{created}</div>
        </div>""", unsafe_allow_html=True)


def _render_plain_text(text: str) -> None:
    """Render plain-text output (logs, exec, prune results) cleanly."""
    # Success banner
    if text.startswith("✅"):
        lines = text.split("\n")
        rest  = "\n".join(lines[1:]).strip()
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                    padding:10px 14px;margin-bottom:8px">
            <div style="color:#16a34a;font-weight:700">{lines[0]}</div>
            {"<pre style='margin:8px 0 0;font-size:.78rem;color:#374151;white-space:pre-wrap'>" + rest + "</pre>" if rest else ""}
        </div>""", unsafe_allow_html=True)
    elif text.startswith("❌"):
        st.markdown(f"""
        <div style="background:#fff1f2;border:1px solid #fca5a5;border-radius:8px;
                    padding:10px 14px">
            <div style="color:#dc2626;font-weight:700;white-space:pre-wrap">{text[:2000]}</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.code(text[:6000], language="text")


# ─── Smart result dispatcher ──────────────────────────────────────────────────

def _render_result(tool_name: str, result: str) -> None:
    """Parse result and dispatch to the right beautiful renderer."""
    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        _render_plain_text(result)
        return

    # List results
    if isinstance(data, list) and data:
        first = data[0] if isinstance(data[0], dict) else {}
        if "tags" in first and "size" in first:
            _render_image_list(data); return
        if "status" in first and "image" in first:
            _render_container_list(data); return
        if "mountpoint" in first or "driver" in first and "labels" in first:
            _render_volume_list(data); return
        if "driver" in first and "scope" in first:
            _render_network_list(data); return
        if "created_by" in first:
            _render_history(data); return

    # Single dict results
    if isinstance(data, dict):
        if "cpu_percent" in data:
            _render_stats(data); return
        if "docker_version" in data or "containers_running" in data:
            _render_system_info(data); return
        if "id" in data and ("tags" in data or "status" in data):
            _render_inspect(data); return

    # Fallback: pretty JSON
    st.json(data)


# ─── Tool event renderer ──────────────────────────────────────────────────────

def render_tool_event(ev: ToolEvent):
    if ev.result is None:
        with st.expander(f"🔧 `{ev.tool_name}` — calling…", expanded=True):
            if ev.args:
                st.json(ev.args)
        return

    icon  = "✅" if not ev.error else "❌"
    label = f"{icon} `{ev.tool_name}`"

    with st.expander(label, expanded=True):
        # Args pill row
        if ev.args:
            args_html = " ".join(
                f'<span style="background:#f1f5f9;border:1px solid #e2e8f0;'
                f'padding:2px 8px;border-radius:4px;font-size:.72rem;font-family:monospace;'
                f'color:#334155"><b>{k}</b>: {str(v)[:40]}</span>'
                for k, v in ev.args.items()
            )
            st.markdown(f'<div style="margin-bottom:10px">{args_html}</div>',
                        unsafe_allow_html=True)

        # Render result beautifully
        _render_result(ev.tool_name, ev.result)


def render_message(msg: ChatMessage):
    if msg.role == "user":
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant", avatar="🐳"):
            for ev in msg.tool_events:
                render_tool_event(ev)
            if msg.content:
                st.markdown(msg.content)

# ─── Main ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="dock-header"><span style="font-size:2rem">🐳</span>'
    '<h1>Docker MCP</h1>'
    '<span class="dock-badge">Gemini 2.5 Flash · Vertex AI</span></div>',
    unsafe_allow_html=True,
)

for msg in st.session_state.messages:
    render_message(msg)

pending    = st.session_state.pop("pending_prompt", None)
user_input = st.chat_input("Ask anything about your Docker environment…",
                            disabled=st.session_state.processing)
prompt = pending or user_input

if prompt and not st.session_state.processing:
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🐳"):
        status_ph = st.empty()
        tools_ph  = st.empty()
        text_ph   = st.empty()

        def _status(html: str):
            status_ph.markdown(
                f'<div style="color:#0284c7;font-style:italic;font-size:.88rem">{html}</div>',
                unsafe_allow_html=True,
            )

        _status("🤔 Gemini is thinking…")

        eq           = _queue.Queue()
        history_copy = list(st.session_state.gemini_history)
        box: dict    = {}

        def _thread():
            box["h"] = run_agent_sync(prompt, history_copy, eq)

        threading.Thread(target=_thread, daemon=True).start()

        asst_msg   = ChatMessage(role="assistant", content="")
        cur_ev: ToolEvent | None = None
        final_text = ""

        while True:
            try:
                kind, payload = eq.get(timeout=0.15)
            except _queue.Empty:
                continue

            if kind == "done":
                break
            elif kind == "error":
                status_ph.error(f"❌ {payload}")
                break
            elif kind == "thinking":
                _status("🤔 Gemini is thinking…")
            elif kind == "text":
                final_text = payload
                status_ph.empty()
                text_ph.markdown(payload)
            elif kind == "tool_call":
                _status(f'🔧 Calling <code>{payload["name"]}</code>…')
                ev = ToolEvent(tool_name=payload["name"], args=payload["args"])
                asst_msg.tool_events.append(ev)
                cur_ev = ev
                with tools_ph.container():
                    for e in asst_msg.tool_events:
                        render_tool_event(e)
            elif kind == "tool_result":
                if cur_ev and cur_ev.tool_name == payload["name"]:
                    cur_ev.result = payload["result"]
                    cur_ev.error  = payload["error"]
                _status("⚙️ Processing result…")
                with tools_ph.container():
                    for e in asst_msg.tool_events:
                        render_tool_event(e)

        status_ph.empty()
        asst_msg.content = final_text
        st.session_state.messages.append(asst_msg)
        st.session_state.gemini_history = box.get("h", history_copy)
        with tools_ph.container():
            for ev in asst_msg.tool_events:
                render_tool_event(ev)
        text_ph.markdown(final_text)

# ─── Empty state ─────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
<div style="text-align:center;padding:64px 20px;color:#64748b">
    <div style="font-size:4rem;margin-bottom:16px">🐳</div>
    <h3 style="color:#0f172a;margin-bottom:8px;font-weight:800">Docker MCP Assistant</h3>
    <p style="max-width:500px;margin:0 auto;line-height:1.7;color:#475569">
        Ask anything in plain English — pull images, run containers,
        tail logs, inspect stats, manage volumes &amp; networks,
        build from a Dockerfile, and more.
    </p>
    <div style="margin-top:32px;display:flex;flex-wrap:wrap;gap:10px;justify-content:center">
        <span style="background:#e0f2fe;border:1px solid #bae6fd;color:#0369a1;
                     padding:8px 16px;border-radius:20px;font-size:.83rem;font-weight:500">
            Pull nginx:alpine and run it on port 8080
        </span>
        <span style="background:#e0f2fe;border:1px solid #bae6fd;color:#0369a1;
                     padding:8px 16px;border-radius:20px;font-size:.83rem;font-weight:500">
            Show CPU &amp; memory stats for all containers
        </span>
        <span style="background:#e0f2fe;border:1px solid #bae6fd;color:#0369a1;
                     padding:8px 16px;border-radius:20px;font-size:.83rem;font-weight:500">
            Tail last 50 logs from my api container
        </span>
        <span style="background:#e0f2fe;border:1px solid #bae6fd;color:#0369a1;
                     padding:8px 16px;border-radius:20px;font-size:.83rem;font-weight:500">
            Prune all stopped containers &amp; dangling images
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
