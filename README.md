# AgentDesk

A conversational AI agent built with LangChain that:
- remembers the conversation across turns (session-based memory),
- decides on its own whether to answer directly, call a tool, or retrieve
  from a document knowledge base,
- shows its reasoning ("which tool did it use?") in the UI,
- runs as a Streamlit app.

Built with Gemini (via `langchain-google-genai`), ChromaDB, and local
HuggingFace embeddings — no paid embedding API required.

---

## Architecture

```
User message (Streamlit chat_input)
        │
        ▼
RunnableWithMessageHistory   ← loads/saves chat_history for this session_id
        │                       (agent/memory.py)
        ▼
AgentExecutor (tool-calling agent, Gemini)
        │
        ├── answers directly from the LLM, OR
        ├── calls calculator / current_datetime / product_lookup, OR
        └── calls search_documents
                    │
                    ▼
            Chroma retriever (agent/vectorstore.py)
            HuggingFace all-MiniLM-L6-v2 embeddings
            persisted in ./chroma_db
        │
        ▼
Response + intermediate_steps (tool calls made)
        │
        ▼
Streamlit renders the answer, and the tool trace in an expander
```

```
agentdesk/
├── app.py                      # Streamlit UI (chat, memory controls, export, tool trace)
├── agent/
│   ├── agent.py                 # LLM + AgentExecutor + memory wiring
│   ├── tools.py                 # calculator, datetime, product_lookup, search_documents
│   ├── memory.py                # session-based chat history store
│   └── vectorstore.py           # ChromaDB + HuggingFace embeddings + ingestion
├── data/
│   ├── documents/                # sample .txt knowledge base (policies, FAQs, onboarding)
│   └── products.csv              # backs the custom product_lookup tool
├── scripts/
│   └── ingest.py                 # one-off / repeatable script to (re)build the vector store
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup (local)

**Requirements:** Python 3.10+

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey),
   then:
   ```bash
   cp .env.example .env
   # edit .env and paste your key into GOOGLE_API_KEY
   ```

3. Build the vector store from the sample documents (first run only, or
   whenever you edit `data/documents/`):
   ```bash
   python scripts/ingest.py
   ```
   (You can also do this from inside the app — the sidebar shows an
   "Ingest sample documents" button if the store is empty.)

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## Example queries to test routing

| Question | Expected path |
|---|---|
| "What's the capital of France?" | Direct LLM answer, no tool |
| "What's 18% of 245?" | `calculator` tool |
| "What's today's date?" | `current_datetime` tool |
| "Do you sell a standing desk? How much?" | `product_lookup` tool |
| "How many days a week can I work remotely?" | `search_documents` (remote work policy) |
| "What did I just ask you?" | Answered from `chat_history`, no tool — tests memory |

Each response's expander shows exactly which tool (if any) fired, with its
input and output, so you can verify routing at a glance.

---

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (see note on commit history below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app."
3. Point it at your repo, branch `main`, main file path `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_key_here"
   GEMINI_MODEL = "gemini-2.5-flash"
   ```
   (Streamlit Cloud injects these as environment variables, so
   `os.getenv("GOOGLE_API_KEY")` in `agent.py` picks them up the same way
   it reads a local `.env` — no code changes needed.)
5. Deploy. On first load, use the sidebar's "Ingest sample documents"
   button once — Streamlit Cloud's filesystem is ephemeral/per-instance,
   so the Chroma store is rebuilt fresh (it's small and fast, so this is a
   non-issue for a demo-sized knowledge base).

---

## Design decisions

**Memory: `RunnableWithMessageHistory`, not `ConversationChain`/legacy
memory classes.** LangChain has deprecated the old memory abstractions in
favor of treating history as an explicit input/output the runnable reads
and writes, keyed by a session id. This is more explicit about what state
exists and where, and is the pattern LangChain's current docs recommend.

**Agent framework: classic `AgentExecutor`, not LangGraph.** LangGraph is
the better choice for multi-step/cyclic/multi-agent control flow, but this
project is a single flat-toolset agent with no branching logic beyond
"pick a tool or don't." `AgentExecutor` covers that in less code and is
easier to read top-to-bottom in one file. If this grew to need
human-in-the-loop approval before a tool call, parallel sub-agents, or
conditional routing between multiple specialized agents, LangGraph would
be the right next step.

**Embeddings: local HuggingFace (`all-MiniLM-L6-v2`), not an API.** This
keeps the retrieval half of the app completely free and runnable offline
— no billing setup required just to try the project. The tradeoff is
somewhat lower embedding quality than a top-tier hosted model; for a small,
topically narrow document set (a handful of internal policy docs) that
tradeoff is negligible in practice.

**Chunking: `RecursiveCharacterTextSplitter`, 500 chars / 50 overlap.**
Large enough to keep a paragraph's context together (most answers in the
sample docs live in 1-2 paragraphs), small enough to keep retrieved chunks
focused and avoid diluting the LLM's context with irrelevant neighboring
policy sections.

**Tools: two "no API key" utility tools (calculator, datetime) plus one
custom data-backed tool (`product_lookup` over a local CSV), plus document
retrieval as a fourth tool.** The calculator and datetime tools are
included because "did the agent correctly avoid using an LLM's fuzzy
internal math/date sense" is a very visible, easy-to-verify correctness
check. `product_lookup` demonstrates wiring in a first-party data source
with its own parsing and error handling (missing file, no matches), rather
than just calling a third-party API wrapper. Wrapping the Chroma retriever
as a tool (rather than always retrieving) lets the agent decide *whether*
retrieval is relevant per question, which is the actual point of Phase 2.

**Error handling.** `AgentExecutor(handle_parsing_errors=True)` prevents a
malformed tool call from crashing the run. Each tool independently
try/excepts its own failure mode (missing CSV, bad expression, empty
Chroma result) and returns a descriptive string instead of raising, so the
agent can explain the failure to the user instead of the whole chain
erroring out. `app.py` also wraps the top-level `agent.invoke()` call so
an unexpected error (e.g. API quota, network issue) shows a clean message
in the chat instead of a Streamlit stack trace.

---

## Known limitations / next steps

- Chroma re-ingestion isn't deduplicated — re-running `ingest.py` on the
  same files adds duplicate chunks. Fine for a demo; a production version
  would hash file content into deterministic chunk IDs.
- Sample knowledge base is `.txt` only; `vectorstore.py`'s loader can be
  extended to PDFs (`PyPDFLoader`) with minimal changes.
- Memory is in-process (a Python dict) — restarting the app clears all
  sessions. Swappable for `RedisChatMessageHistory` or
  `SQLChatMessageHistory` without touching `agent.py` or `app.py`.
