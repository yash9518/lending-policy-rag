"""Load policy documents, split them into clause-level chunks, and index them.

Chunking is structure-aware: policy documents are written as numbered
sections ("1. ELIGIBILITY CRITERIA:") containing bullet clauses ("- ...").
Each bullet becomes one chunk tagged with its section, so a retrieved chunk
is always a complete rule with a stable citation id such as
``RLG §1.2`` (document code + section.clause). Text that does not follow this layout
falls back to paragraph/size-based splitting.

Run ``python -m src.ingest`` to (re)build the dense vector index. Rebuilding
drops the old collection first, so repeated runs never duplicate chunks.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config

SECTION_RE = re.compile(r"^\s*(\d+)\.\s+(.+?):?\s*$")
BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*)$")
FALLBACK_CHUNK_CHARS = 600


def doc_code(source: str) -> str:
    """Short citation prefix from the file name: retail_lending_guidelines -> RLG."""
    words = [w for w in re.split(r"[_\-\s]+", Path(source).stem) if w and w.lower() not in {"and", "of", "the"}]
    return "".join(w[0] for w in words).upper() or "DOC"


@dataclass
class Chunk:
    id: str  # citation id, e.g. "RLG §1.2" (document code + section.clause)
    text: str  # clause text shown to users and the LLM
    source: str  # file name
    section: str  # section title, e.g. "Eligibility Criteria"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def indexed_text(self) -> str:
        """Text used for embedding / keyword search (adds section context)."""
        return f"{self.section}: {self.text}" if self.section else self.text


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def _split_long(text: str, max_chars: int = FALLBACK_CHUNK_CHARS) -> list[str]:
    sentences = re.split(r"(?<=[.;])\s+", text)
    parts, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = ""
        current += " " + sentence
    if current.strip():
        parts.append(current.strip())
    return parts


def chunk_document(text: str, source: str) -> list[Chunk]:
    stem = doc_code(source)
    chunks: list[Chunk] = []
    section_no, section_title = "0", "Document Header"
    clause_no = 0
    loose: list[str] = []  # lines that are neither headings nor bullets

    def flush_loose():
        nonlocal clause_no
        paragraph = " ".join(line.strip() for line in loose if line.strip())
        loose.clear()
        if not paragraph:
            return
        for part in _split_long(paragraph):
            clause_no += 1
            chunks.append(Chunk(f"{stem} §{section_no}.{clause_no}", part, source, section_title))

    for line in text.splitlines():
        if not line.strip():
            flush_loose()
            continue
        section = SECTION_RE.match(line)
        bullet = BULLET_RE.match(line)
        if section and not bullet:
            flush_loose()
            section_no = section.group(1)
            section_title = section.group(2).strip().rstrip(":").title()
            clause_no = 0
        elif bullet:
            flush_loose()
            clause_no += 1
            chunks.append(Chunk(f"{stem} §{section_no}.{clause_no}", bullet.group(1).strip(), source, section_title))
        else:
            loose.append(line)
    flush_loose()
    return chunks


def load_chunks(policy_dir: Path = config.POLICY_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(policy_dir.iterdir()):
        if path.suffix.lower() in {".txt", ".md", ".pdf"}:
            chunks.extend(chunk_document(read_document(path), path.name))
    ids = [c.id for c in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate clause ids {sorted(duplicates)}: two documents share a code; rename one file")
    return chunks


def build_vector_store(policy_dir: Path = config.POLICY_DIR, chroma_dir: Path = config.CHROMA_DIR) -> int:
    """(Re)build the Chroma collection from the policy folder. Returns chunk count."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks = load_chunks(policy_dir)
    if not chunks:
        raise RuntimeError(f"No policy documents found in {policy_dir}")

    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode([c.indexed_text for c in chunks], normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    collection = client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c.source, "section": c.section} for c in chunks],
    )
    print(f"Indexed {len(chunks)} clauses from {policy_dir} into {chroma_dir}")
    return len(chunks)


if __name__ == "__main__":
    build_vector_store()
