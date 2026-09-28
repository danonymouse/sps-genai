"""
Shared corpus-analysis helpers used as *tools* across Module 3 labs.

These wrap the students' existing Module 2 `BigramModel` so their agents can
call familiar functions. Keeping tools pure and side-effect-free makes them
easy for a language model to use correctly.

Design principle for tools (Anthropic 2024, "Prompt-engineering your tools"):
- One obvious job per tool.
- Docstrings written FOR the model — they become the tool description.
- Fail loudly and structurally (return an error dict, don't raise).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TypedDict


class ToolError(TypedDict):
    error: str


# --- helpers ----------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphabetic-token split. Deliberately simple so students can
    see exactly what the tool does."""
    return re.findall(r"[a-zA-Z']+", text.lower())


# --- tools exposed to agents ------------------------------------------------


def word_frequency(corpus: list[str], word: str) -> dict:
    """Return how many times `word` appears in the corpus (case-insensitive).

    Args:
        corpus: list of sentences/documents.
        word: the word to count.

    Returns:
        {"word": str, "count": int, "total_tokens": int}
    """
    if not word:
        return {"error": "word must be a non-empty string"}
    tokens = [t for doc in corpus for t in _tokenize(doc)]
    return {
        "word": word.lower(),
        "count": tokens.count(word.lower()),
        "total_tokens": len(tokens),
    }


def top_bigrams(corpus: list[str], n: int = 5) -> dict:
    """Return the `n` most common bigrams in the corpus.

    Args:
        corpus: list of sentences/documents.
        n: how many bigrams to return (1 <= n <= 50).

    Returns:
        {"bigrams": [[w1, w2, count], ...]}
    """
    if not (1 <= n <= 50):
        return {"error": "n must be between 1 and 50"}
    tokens = [t for doc in corpus for t in _tokenize(doc)]
    pairs = Counter(zip(tokens, tokens[1:]))
    return {"bigrams": [[w1, w2, c] for (w1, w2), c in pairs.most_common(n)]}


def next_word_options(corpus: list[str], word: str, top_k: int = 3) -> dict:
    """Return the top `top_k` words that most often follow `word`.

    Great for demonstrating: "the agent asks the model which word likely
    follows 'the', and gets probabilistic answers grounded in the corpus."
    """
    if not word:
        return {"error": "word must be a non-empty string"}
    tokens = [t for doc in corpus for t in _tokenize(doc)]
    following = Counter(
        b for a, b in zip(tokens, tokens[1:]) if a == word.lower()
    )
    if not following:
        return {"word": word.lower(), "options": []}
    return {
        "word": word.lower(),
        "options": [[w, c] for w, c in following.most_common(top_k)],
    }


def calc(expression: str) -> dict:
    """Safely evaluate a simple arithmetic expression (numbers and + - * / ( )).

    Included as a canonical "capability extension" tool — LLMs are bad at math,
    a calculator fixes that in one line.
    """
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression or ""):
        return {"error": "only digits and + - * / ( ) . allowed"}
    try:
        # Restricted eval: no names, no builtins.
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as e:  # noqa: BLE001
        return {"error": f"eval failed: {e}"}
    return {"expression": expression, "result": result}


# --- default corpus used in labs -------------------------------------------

DEFAULT_CORPUS: list[str] = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
    "word embeddings map words to dense vectors so that similar words end up close together",
    "the transformer architecture powers most modern language models",
]


# JSON-schema definitions students hand to the LLM. Kept next to the
# implementations so they stay in sync.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "word_frequency",
            "description": "Count how many times a specific word appears in the current corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "The word to count."}
                },
                "required": ["word"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "top_bigrams",
            "description": "Return the most common word-pairs (bigrams) in the current corpus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "How many bigrams to return (1-50). Default 5.",
                        "minimum": 1,
                        "maximum": 50,
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "next_word_options",
            "description": "Given a word, return the words most likely to follow it in the corpus, "
            "with counts. Useful for exploring bigram probabilities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["word"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a simple arithmetic expression. Use this whenever the user asks "
            "for a numeric computation — do NOT compute mentally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression using digits and + - * / ( ) . only.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]


# Dispatch table used by the ReAct loop in Lecture 1.
# Small open models (llama3.x, qwen2.5) sometimes emit stringified numbers even
# when the schema declares `integer`. We coerce defensively here so the tools
# work identically on hosted (gpt-4o-mini) and local (Ollama) models.
def _as_int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


TOOL_IMPLS = {
    "word_frequency": lambda args, corpus: word_frequency(corpus, str(args.get("word", ""))),
    "top_bigrams": lambda args, corpus: top_bigrams(corpus, _as_int(args.get("n"), 5)),
    "next_word_options": lambda args, corpus: next_word_options(
        corpus, str(args.get("word", "")), _as_int(args.get("top_k"), 3)
    ),
    "calc": lambda args, _corpus: calc(str(args.get("expression", ""))),
}
