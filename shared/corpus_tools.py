import ast
import re
from collections import Counter

DEFAULT_CORPUS = """
The quick brown fox jumps over the lazy dog.
The dog barks at the fox.
The fox and the dog count the stars.
The count of the fox is one.
The lazy dog and the quick fox sleep.
""".strip()


def _tokenize(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def top_bigrams(args, corpus):
    """Return the top n bigrams with counts."""
    n = int(args.get("n", 3))
    tokens = _tokenize(corpus)
    bigrams = Counter(tuple(tokens[i:i + 2]) for i in range(len(tokens) - 1))
    ranked = [
        {"bigram": " ".join(bigram), "count": count}
        for bigram, count in bigrams.most_common(n)
    ]
    return {"top_bigrams": ranked}


def word_frequency(args, corpus):
    """Count a single word in the corpus."""
    word = str(args.get("word", "")).lower()
    tokens = _tokenize(corpus)
    return {"word": word, "count": sum(1 for token in tokens if token == word)}


def calc(args, corpus):
    """Safely evaluate an arithmetic expression containing only numbers and operators."""
    expr = str(args.get("expression", ""))
    if not expr:
        raise ValueError("Expression is required.")

    if re.search(r"[A-Za-z_]", expr):
        raise ValueError("Only arithmetic expressions are allowed.")

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression: {expr}") from exc

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Constant,
    )
    for subnode in ast.walk(node):
        if not isinstance(subnode, allowed_nodes):
            raise ValueError("Only arithmetic is allowed.")

    result = eval(compile(node, "<expr>", "eval"), {"__builtins__": {}}, {})
    return {"expression": expr, "result": result}


TOOL_IMPLS = {
    "top_bigrams": top_bigrams,
    "word_frequency": word_frequency,
    "calc": calc,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "top_bigrams",
            "description": "Return the most common bigrams in the corpus.",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer", "description": "Number of bigrams to return."}},
                "required": ["n"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "word_frequency",
            "description": "Count how many times a word appears in the corpus.",
            "parameters": {
                "type": "object",
                "properties": {"word": {"type": "string", "description": "Word to count."}},
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Evaluate a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression like 2 + 3 * 4"}
                },
                "required": ["expression"],
            },
        },
    },
]
