"""
Ingestion + retrieval helpers for Lecture 3 (RAG agents).

Design goals:
- **Free / local by default.** Uses `sentence-transformers` (all-MiniLM-L6-v2) for
  embeddings and a local, on-disk Chroma collection. No paid API key needed just
  to do retrieval.
- **Provenance-friendly.** Every chunk carries `{source, chunk_index}` metadata so
  the agent can cite `[source: filename#chunkN]`.
- **Simple, readable chunking** so students can see exactly what happens.

Public API
----------
build_index(source_dir, collection="course_readings", ...) -> int   # returns #chunks
search(query, k=5, collection="course_readings") -> list[dict]
reset_collection(collection="course_readings") -> None
list_sources(collection="course_readings") -> list[str]

Each `search` result is a dict:
    {
        "text": str,          # the chunk text
        "source": str,        # filename it came from
        "chunk_index": int,   # position within that file
        "citation": str,      # ready-to-use "filename#chunkN"
        "distance": float,    # lower = more similar
    }
"""

from __future__ import annotations

from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = MODULE_ROOT / ".chroma"

_SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


# --- document loading -------------------------------------------------------


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF. Requires `pypdf`."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pypdf is required to read PDFs. `pip install pypdf` or convert to .md."
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_documents(source_dir: Path) -> list[tuple[str, str]]:
    """Return a list of (source_name, full_text) for every supported file."""
    docs: list[tuple[str, str]] = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue
        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append((path.name, text))
    return docs


# --- chunking ---------------------------------------------------------------


def _chunk(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Word-based sliding-window chunking.

    We chunk on words (not characters) so chunks don't split mid-word, and we
    overlap consecutive chunks so a sentence spanning a boundary still appears
    whole in at least one chunk.
    """
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if window:
            chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


# --- Chroma plumbing --------------------------------------------------------


def _embedding_fn():
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def _client():
    import chromadb

    CHROMA_DIR.mkdir(exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_or_create(collection: str):
    return _client().get_or_create_collection(
        name=collection,
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},
    )


# --- public API -------------------------------------------------------------


def reset_collection(collection: str = "course_readings") -> None:
    """Delete the collection so you can re-ingest from scratch."""
    client = _client()
    try:
        client.delete_collection(collection)
    except Exception:  # noqa: BLE001 — fine if it didn't exist
        pass


def build_index(
    source_dir: str | Path,
    collection: str = "course_readings",
    *,
    chunk_size: int = 800,
    overlap: int = 150,
    reset: bool = True,
) -> int:
    """Chunk, embed, and store every supported file under `source_dir`.

    Returns the number of chunks indexed.
    """
    source_dir = Path(source_dir)
    if not source_dir.is_absolute():
        source_dir = MODULE_ROOT / source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir not found: {source_dir}")

    if reset:
        reset_collection(collection)
    coll = _get_or_create(collection)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for source_name, text in _read_documents(source_dir):
        for idx, chunk in enumerate(_chunk(text, chunk_size, overlap)):
            ids.append(f"{source_name}#chunk{idx}")
            documents.append(chunk)
            metadatas.append({"source": source_name, "chunk_index": idx})

    if not documents:
        raise RuntimeError(
            f"No supported documents (.md/.txt/.pdf) found in {source_dir}."
        )

    # Chroma embeds via the collection's embedding function on add.
    coll.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(documents)


def search(
    query: str,
    k: int = 5,
    collection: str = "course_readings",
) -> list[dict]:
    """Vector-search the collection. Returns a list of provenance-tagged hits."""
    coll = _get_or_create(collection)
    if coll.count() == 0:
        return []
    res = coll.query(query_texts=[query], n_results=min(k, coll.count()))

    hits: list[dict] = []
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for text, meta, dist in zip(docs, metas, dists):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", -1)
        hits.append(
            {
                "text": text,
                "source": source,
                "chunk_index": chunk_index,
                "citation": f"{source}#chunk{chunk_index}",
                "distance": float(dist),
            }
        )
    return hits


def list_sources(collection: str = "course_readings") -> list[str]:
    """Return the distinct source filenames currently indexed."""
    coll = _get_or_create(collection)
    if coll.count() == 0:
        return []
    got = coll.get(include=["metadatas"])
    return sorted({m.get("source", "unknown") for m in got.get("metadatas", [])})


if __name__ == "__main__":
    # Tiny smoke test against the bundled readings.
    n = build_index("data/readings")
    print(f"indexed {n} chunks from {list_sources()}")
    for hit in search("when should I use an agent instead of a workflow?", k=3):
        print(f"  [{hit['citation']}] ({hit['distance']:.3f}) {hit['text'][:90]}...")
