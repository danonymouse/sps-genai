"""
A trivial append-only scratchpad the agent uses as short-term memory across turns.

Kept deliberately simple (a Markdown file) so students can open `notes.md` and
literally watch the agent's memory grow. Every function returns a dict so it's
easy to expose as an agent tool.

Public API
----------
save_note(text, path=DEFAULT_NOTES_PATH) -> dict
read_notes(path=DEFAULT_NOTES_PATH) -> dict
clear_notes(path=DEFAULT_NOTES_PATH) -> dict
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTES_PATH = MODULE_ROOT / "notes.md"


def save_note(text: str, path: str | Path = DEFAULT_NOTES_PATH) -> dict:
    """Append a timestamped note to the scratchpad.

    Args:
        text: the finding to remember (one or two sentences works best).

    Returns:
        {"saved": True, "note_count": int}
    """
    path = Path(path)
    if not text or not text.strip():
        return {"error": "note text must be non-empty"}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"- [{timestamp}] {text.strip()}\n")
    return {"saved": True, "note_count": _count_notes(path)}


def read_notes(path: str | Path = DEFAULT_NOTES_PATH) -> dict:
    """Return everything saved to the scratchpad so far.

    Returns:
        {"notes": str, "note_count": int}
    """
    path = Path(path)
    if not path.exists():
        return {"notes": "", "note_count": 0}
    content = path.read_text(encoding="utf-8")
    return {"notes": content, "note_count": _count_notes(path)}


def clear_notes(path: str | Path = DEFAULT_NOTES_PATH) -> dict:
    """Wipe the scratchpad (useful between lab runs)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    return {"cleared": True}


def _count_notes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("- ["))
