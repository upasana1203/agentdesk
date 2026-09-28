"""
tools.py
--------
All tools the agent can call. Each tool is a small, single-purpose function
decorated with LangChain's `@tool`, which turns the docstring + type hints
into the schema the LLM uses to decide when and how to call it.

Tools included:
  1. calculator        - no API key. Safe arithmetic evaluation (ast-based,
                          NOT eval()) so the agent can do math without
                          hallucinating numbers.
  2. current_datetime   - no API key. Grounds "what's today's date" style
                          questions in a real value instead of the LLM's
                          training-data guess.
  3. product_lookup     - custom tool. Reads data/products.csv and answers
                          questions about a small product catalog. This is
                          the "real engineering" tool: it's our own data
                          source, our own parsing/error handling, not just
                          a wrapped third-party API.
  4. search_documents   - wraps the Chroma vector store (see vectorstore.py)
                          as a tool, so document retrieval is just another
                          action the agent can choose, not a separate code
                          path.
"""

import ast
import operator
import os
from datetime import datetime

import pandas as pd
from langchain_core.tools import tool

from agent.vectorstore import get_retriever

# ---------------------------------------------------------------------------
# 1. Calculator (safe, no eval())
# ---------------------------------------------------------------------------

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '12 * (3 + 4) / 2'.
    Supports +, -, *, /, **, %% and parentheses. Use this instead of doing
    math yourself so the result is exact, not estimated."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:
        return f"Error: could not evaluate '{expression}' ({exc})"


# ---------------------------------------------------------------------------
# 2. Current date/time
# ---------------------------------------------------------------------------

@tool
def current_datetime(_: str = "") -> str:
    """Return the current date and time. Use this for any question about
    today's date, the current year, or elapsed-time calculations. Input is
    ignored."""
    return datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")


# ---------------------------------------------------------------------------
# 3. Custom tool: CSV-backed product lookup
# ---------------------------------------------------------------------------

_PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "products.csv")
_products_df: pd.DataFrame | None = None


def _load_products() -> pd.DataFrame:
    global _products_df
    if _products_df is None:
        _products_df = pd.read_csv(_PRODUCTS_PATH)
    return _products_df


@tool
def product_lookup(query: str) -> str:
    """Look up product info (name, price, category, stock) from the internal
    product catalog by product name or category keyword. Use this for any
    question about what products are sold, their price, or availability.
    Example query: 'wireless mouse' or 'electronics'."""
    try:
        df = _load_products()
    except FileNotFoundError:
        return "Error: product catalog file not found."

    query_lower = query.strip().lower()
    mask = (
        df["name"].str.lower().str.contains(query_lower, na=False)
        | df["category"].str.lower().str.contains(query_lower, na=False)
    )
    matches = df[mask]

    if matches.empty:
        return f"No products found matching '{query}'."

    lines = [
        f"- {row['name']} | ${row['price']:.2f} | {row['category']} | "
        f"{'in stock' if row['in_stock'] else 'out of stock'}"
        for _, row in matches.iterrows()
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Document retrieval (Chroma) as a tool
# ---------------------------------------------------------------------------

@tool
def search_documents(query: str) -> str:
    """Search the internal knowledge base (company policy docs, FAQs,
    onboarding guides) for information relevant to the query. Use this when
    the user asks something that sounds like it depends on internal/company
    documents rather than general knowledge."""
    retriever = get_retriever()
    docs = retriever.invoke(query)

    if not docs:
        return "No relevant documents found in the knowledge base for this query."

    chunks = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "unknown")
        chunks.append(f"[{i}] (source: {source})\n{d.page_content}")
    return "\n\n".join(chunks)


def get_all_tools():
    """Convenience accessor used by agent.py to build the agent's tool list."""
    return [calculator, current_datetime, product_lookup, search_documents]
