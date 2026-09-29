"""
app.py
------
Streamlit front-end for the AgentDesk conversational agent.

Responsibilities kept deliberately thin here: all agent/tool/memory/
vectorstore logic lives in agent/*.py. This file is just UI + wiring.
"""

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.agent import build_agent_with_memory  # noqa: E402
from agent.memory import clear_session_history  # noqa: E402
from agent.vectorstore import collection_is_empty, ingest_documents  # noqa: E402

st.set_page_config(page_title="AgentDesk", page_icon="🧠", layout="centered")

# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "display_messages" not in st.session_state:
    # separate from LangChain's internal chat_history -- this list also
    # carries the tool trace metadata for rendering
    st.session_state.display_messages = []

if "agent" not in st.session_state:
    try:
        st.session_state.agent = build_agent_with_memory()
        st.session_state.agent_error = None
    except Exception as exc:  # e.g. missing API key
        st.session_state.agent = None
        st.session_state.agent_error = str(exc)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("AgentDesk 🧠")
    st.caption("LangChain agent · Gemini · ChromaDB · Streamlit")

    st.divider()
    st.subheader("Knowledge base")
    if collection_is_empty():
        st.warning("Vector store is empty.")
        if st.button("Ingest sample documents"):
            with st.spinner("Chunking and embedding documents..."):
                try:
                    n = ingest_documents()
                    st.success(f"Ingested {n} chunks.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Ingestion failed: {exc}")
    else:
        st.success("Knowledge base loaded.")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        clear_session_history(st.session_state.session_id)
        st.session_state.display_messages = []
        st.rerun()

    if st.session_state.display_messages:
        export_payload = json.dumps(
            {
                "session_id": st.session_state.session_id,
                "exported_at": datetime.now().isoformat(),
                "messages": st.session_state.display_messages,
            },
            indent=2,
        )
        st.download_button(
            "⬇️ Export conversation (JSON)",
            data=export_payload,
            file_name=f"agentdesk_conversation_{st.session_state.session_id[:8]}.json",
            mime="application/json",
        )

    st.divider()
    st.caption(
        "Try: 'What's 18% of 245?' · 'What's today's date?' · "
        "'Do you sell a standing desk?' · "
        "'How many days can I work internationally?'"
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("AgentDesk")

if st.session_state.agent_error:
    st.error(
        f"Agent could not be initialized: {st.session_state.agent_error}\n\n"
        "Copy .env.example to .env and add your GOOGLE_API_KEY, then restart the app."
    )
    st.stop()

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tool_trace"):
            with st.expander("🔧 Reasoning trace (tools used)"):
                for step in msg["tool_trace"]:
                    st.markdown(f"**Tool:** `{step['tool']}`")
                    st.markdown(f"**Input:** `{step['input']}`")
                    st.markdown(f"**Output:** {step['output']}")
                    st.markdown("---")

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.agent.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": st.session_state.session_id}},
                )
                answer = result.get("output", "Sorry, I didn't get a response.")
                if isinstance(answer, list):
                    answer = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in answer
                    )

                tool_trace = []
                for action, observation in result.get("intermediate_steps", []):
                    tool_trace.append(
                        {
                            "tool": action.tool,
                            "input": action.tool_input,
                            "output": str(observation)[:1000],  # keep the UI readable
                        }
                    )

                st.markdown(answer)
                if tool_trace:
                    with st.expander("🔧 Reasoning trace (tools used)"):
                        for step in tool_trace:
                            st.markdown(f"**Tool:** `{step['tool']}`")
                            st.markdown(f"**Input:** `{step['input']}`")
                            st.markdown(f"**Output:** {step['output']}")
                            st.markdown("---")

                st.session_state.display_messages.append(
                    {"role": "assistant", "content": answer, "tool_trace": tool_trace}
                )

            except Exception as exc:
                error_msg = f"Something went wrong handling that request: {exc}"
                st.error(error_msg)
                st.session_state.display_messages.append(
                    {"role": "assistant", "content": error_msg, "tool_trace": []}
                )

