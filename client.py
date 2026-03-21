import asyncio
import logging
from pathlib import Path
from typing import List, Tuple

from vertexai.generative_models import (
    GenerativeModel,
    Tool,
    FunctionDeclaration,
    Content,
    Part,
)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"
MAX_TOOL_ROUNDS = 8
MAX_PARALLEL_TIMEOUT = 30  # seconds
MAX_RETRIES = 2

MCP_SERVER_FILE = "server.py"

DESTRUCTIVE_KEYWORDS = [
    "remove",
    "delete",
    "prune",
    "force",
    "rm",
    "stop",
]

SYSTEM_PROMPT = """
You are a senior DevOps automation agent.

You control Docker infrastructure through structured tools.

Rules:
1. Always use tools for infrastructure operations.
2. Chain multiple tools when required.
3. Execute independent read-only operations in parallel.
4. Never hallucinate Docker state.
5. For destructive actions (remove, prune, force stop):
   - Ask for confirmation before executing.
6. If a tool fails:
   - Clearly explain why.
   - Suggest a concrete fix.
7. Be concise and operational.
8. Prefer safe and idempotent actions.
9. Do not guess parameters.
You are an infrastructure operator, not a chatbot.
"""


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("enterprise-agent")


# ─────────────────────────────────────────────
# Safety Guard
# ─────────────────────────────────────────────

def is_destructive(tool_name: str) -> bool:
    tool_name = tool_name.lower()
    return any(keyword in tool_name for keyword in DESTRUCTIVE_KEYWORDS)


# ─────────────────────────────────────────────
# Tool Execution (Parallel + Retry + Timeout)
# ─────────────────────────────────────────────

async def execute_tool_with_retry(
    session: ClientSession,
    call,
) -> Tuple[str, str]:

    tool_name = call.name
    args = dict(call.args or {})

    logger.info(f"Calling tool: {tool_name} | args={args}")

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, args),
                timeout=MAX_PARALLEL_TIMEOUT,
            )

            result_text = "\n".join(
                block.text for block in result.content
                if hasattr(block, "text")
            ) or "(no output)"

            logger.info(f"Tool success: {tool_name}")
            return tool_name, result_text

        except Exception as e:
            logger.warning(
                f"Tool failed: {tool_name} | attempt {attempt+1} | error={e}"
            )

            if attempt >= MAX_RETRIES:
                return tool_name, f"ERROR: {e}"

            await asyncio.sleep(1)


# ─────────────────────────────────────────────
# Agent Loop
# ─────────────────────────────────────────────

async def agent_loop(
    model: GenerativeModel,
    session: ClientSession,
    vertex_tools,
    history: List[Content],
):

    for round_index in range(MAX_TOOL_ROUNDS):

        logger.info(f"Agent round {round_index+1}")

        response = model.generate_content(
            contents=history,
            tools=vertex_tools,
        )

        candidate = response.candidates[0]
        history.append(candidate.content)

        parts = candidate.content.parts

        tool_calls = []
        text_output = []

        for part in parts:
            if hasattr(part, "text") and part.text:
                text_output.append(part.text)

            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append(part.function_call)

        if text_output:
            print("\nAssistant:\n", "".join(text_output))

        if not tool_calls:
            logger.info("No more tool calls. Ending round.")
            break

        # Safety check for destructive operations
        for call in tool_calls:
            if is_destructive(call.name):
                confirm = input(
                    f"⚠ Destructive action detected ({call.name}). Proceed? (yes/no): "
                )
                if confirm.lower() != "yes":
                    print("Operation cancelled.")
                    return

        # Execute tools in parallel
        results = await asyncio.gather(
            *(execute_tool_with_retry(session, call) for call in tool_calls)
        )

        tool_results = [
            Part.from_function_response(
                name=name,
                response={"result": result_text},
            )
            for name, result_text in results
        ]

        # Feed results back
        history.append(Content(role="user", parts=tool_results))


# ─────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────

async def main():

    logger.info("Starting Enterprise Docker MCP Agent")

    model = GenerativeModel(
        MODEL_NAME,
        system_instruction=SYSTEM_PROMPT,
    )

    server_path = Path(MCP_SERVER_FILE).resolve()

    server_params = StdioServerParameters(
        command="python",
        args=[str(server_path)],
    )

    async with stdio_client(server_params) as (stdio, write):
        async with ClientSession(stdio, write) as session:

            await session.initialize()
            logger.info("MCP session initialized")

            tool_list = await session.list_tools()

            function_declarations = []

            for t in tool_list.tools:
                fd = FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=t.inputSchema,
                )
                function_declarations.append(fd)

            vertex_tools = [
                Tool(function_declarations=function_declarations)
            ]

            logger.info(f"{len(function_declarations)} tools loaded")

            history: List[Content] = []

            print("Enterprise Docker Agent Ready.")
            print("Type 'exit' to quit.\n")

            while True:

                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nShutting down.")
                    break

                if user_input.lower() in ("exit", "quit"):
                    break

                if not user_input:
                    continue

                history.append(
                    Content(role="user", parts=[Part.from_text(user_input)])
                )

                try:
                    await agent_loop(
                        model,
                        session,
                        vertex_tools,
                        history,
                    )
                except Exception as e:
                    logger.error(f"Agent error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
