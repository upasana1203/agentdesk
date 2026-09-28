"""
memory.py
---------
Session-based conversational memory.

Design decision: LangChain deprecated the old `ConversationBufferMemory` /
`ConversationChain` classes in favor of the `RunnableWithMessageHistory`
pattern, where history is just a list of messages keyed by a session id and
handed to the runnable on each call. This module owns that session store.

In this project the store is an in-memory Python dict, which is enough for
a local demo / single-process Streamlit Cloud deployment. If you needed
persistence across restarts or multiple users at scale, swap
`InMemoryChatMessageHistory` below for a backed implementation such as
`RedisChatMessageHistory` or `SQLChatMessageHistory` -- the rest of the
app does not need to change, since it only depends on the
`get_session_history` function signature.
"""

from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

# session_id -> BaseChatMessageHistory
_session_store: dict[str, BaseChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Return (and lazily create) the message history for a session id.

    This is the callable `RunnableWithMessageHistory` expects: given a
    session id, it must return an object implementing
    `BaseChatMessageHistory`.
    """
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def clear_session_history(session_id: str) -> None:
    """Wipe a single session's history (used by the 'Clear conversation' button)."""
    if session_id in _session_store:
        _session_store[session_id].clear()


def list_sessions() -> list[str]:
    """Return all known session ids (useful for debugging/admin views)."""
    return list(_session_store.keys())
