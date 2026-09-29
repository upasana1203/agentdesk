"""
agent.py
--------
Builds the tool-calling agent:

  Gemini chat model
    -> create_tool_calling_agent (binds tools + prompt)
    -> AgentExecutor (runs the think/act/observe loop, returns intermediate
       steps so the UI can show a "reasoning trace")
    -> RunnableWithMessageHistory (adds session-based memory on top, per
       memory.py)

Design decision: classic `AgentExecutor` (not LangGraph) was chosen for this
project. LangGraph is the more current, more powerful pattern for complex
agent graphs, but for a single agent with a flat toolset and no branching
control flow, AgentExecutor is less code, easier to read end-to-end in one
file, and is still fully supported. If this project grew multi-agent
routing, human-in-the-loop approval steps, or cyclic tool-calling logic,
LangGraph would be the better fit.
"""

import os

from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.memory import get_session_history
from agent.tools import get_all_tools

load_dotenv()

SYSTEM_PROMPT = """You are AgentDesk, a helpful assistant with access to tools.

Decide for each user message whether to:
  - answer directly from your own knowledge (general knowledge questions),
  - call a tool (calculator, current_datetime, product_lookup) when the
    question needs precise computation, the current date/time, or product
    catalog data, or
  - call search_documents when the question depends on internal/company
    documents (policies, FAQs, onboarding info) rather than general
    knowledge.

Only call a tool when it's actually needed -- don't call search_documents
for general knowledge questions, and don't call product_lookup unless the
question is about the product catalog. If a tool returns no useful result,
say so plainly instead of making something up."""


def _content_to_text(content) -> str:
    """Gemini 3.x can return the answer as a list of content blocks (each with
    a thought signature) instead of a plain string. Flatten it to text so the
    chat history only ever stores clean strings."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type", "text") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _clean_output(result: dict) -> dict:
    result["output"] = _content_to_text(result.get("output", ""))
    return result


def build_agent_executor() -> AgentExecutor:
    api_key = (os.getenv("GOOGLE_API_KEY") or "").strip().strip('"')
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    # Model name per https://ai.google.dev/gemini-api/docs/models -- check
    # that page for the current recommended flash/pro model if this one has
    # been deprecated by the time you run this.
    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        google_api_key=api_key,
    )

    tools = get_all_tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        return_intermediate_steps=True,  # powers the UI's "tool used" trace
        handle_parsing_errors=True,      # don't crash the app on a malformed tool call
        max_iterations=6,
    )


def build_agent_with_memory() -> RunnableWithMessageHistory:
    """The object app.py actually talks to: an agent executor wrapped so
    that chat_history is automatically loaded/saved per session_id."""
    # Clean the output BEFORE the history wrapper saves it, otherwise the raw
    # content blocks get stored as "messages" and the next turn crashes.
    executor = build_agent_executor() | RunnableLambda(_clean_output)
    return RunnableWithMessageHistory(
        executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )