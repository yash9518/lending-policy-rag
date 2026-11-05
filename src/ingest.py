"""Load regulation documents, split them into paragraph-level chunks, and index them.

The corpus is official Reserve Bank of India circulars and master directions
(downloaded by scripts/fetch_sources.py). RBI documents are written as
numbered paragraphs ("2.", "2.1.2", "(i)", "(a)") with annexes, tables and
footnotes. The chunker follows that numbering, so every chunk is one
paragraph (or a paragraph's continuation) with a citation id that matches
the paragraph number in the official text, e.g. ``PEN ¶3(i)`` or
``HFC ¶3(a)``. Table rows stay attached to the paragraph that introduces them.

Run ``python -m src.ingest`` to (re)build the dense vector index. Rebuilding
drops the old collection first, so repeated runs never duplicate chunks.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config

HEADER_SEP = "====="
NAV_RE = re.compile(r"^(?:\d{4}|Archives|All Months|January|February|March|April|May|June|July|August|"
                    r"September|October|November|December)$")
NUM_RE = re.compile(r"^(\d+(?:\.\d+)+)\.?\s+(?=\S)|^(\d+)\.\s+(?=\S)")  # "2.1.3 Text" or "2. Text"
SUB_RE = re.compile(r"^\(([a-z]{1,2}|[ivxl]{1,5})\)\s*(?=\S)")  # "(a) Text" / "(iv) Text"
FOOTNOTE_RE = re.compile(r"^(\d{1,2})\s+(?=[A-Z‘“'\"])")  # "1 As defined in ..."
ANNEX_RE = re.compile(r"^(Annex(?:ure)?)\s*[-–:]?\s*([A-Z0-9IVX]{1,4})?\b(.{0,80})$", re.I)
CHAPTER_RE = re.compile(r"^(Chapter|CHAPTER|Part|PART)\s+([A-Z0-9IVX]{1,5})\b\s*[-–:]?\s*(.{0,120})$")
DELETED_RE = re.compile(r"^\d+\.\s*(?:\d+\s*)?Deleted\.?$")  # "15. 54 Deleted" placeholders
MAX_CHARS = 1600
MIN_CHARS = 160


@dataclass
class Chunk:
    id: str  # citation id, e.g. "PEN ¶3(i)" (document code + paragraph number)
    text: str  # paragraph text shown to users and the LLM
    source: str  # file name
    section: str  # nearest heading, e.g. "Quantum of Loan"
    title: str = ""  # document title
    reference: str = ""  # RBI circular reference, e.g. "RBI/2023-24/53"
    url: str = ""  # official source URL

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def indexed_text(self) -> str:
        """Text used for embedding / keyword search (adds document and section context)."""
        return " · ".join(p for p in (self.title, self.section, self.text) if p)


def doc_code(source: str) -> str:
    """Citation prefix from the file name: PEN.txt -> PEN; retail_lending_guidelines -> RLG."""
    stem = Path(source).stem
    if re.fullmatch(r"[A-Z0-9]{2,6}", stem):
        return stem
    words = [w for w in re.split(r"[_\-\s]+", stem) if w and w.lower() not in {"and", "of", "the"}]
    return "".join(w[0] for w in words).upper() or "DOC"


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8")


def split_header(text: str) -> tuple[dict, str]:
    """Parse the ``KEY: value`` provenance header written by the fetch script."""
    head, sep, body = text.partition(HEADER_SEP)
    if not sep or len(head) > 3000:
        return {}, text
    meta = {}
    for line in head.splitlines():
        key, _, value = line.partition(":")
        if value:
            meta[key.strip().lower()] = value.strip()
    return meta, body


DATE_RE = re.compile(r"^\(?\s*(?:Updated as on\s+)?(?:January|February|March|April|May|June|July|August|September|"
                     r"October|November|December) \d{1,2}, \d{4}")


def _is_heading(line: str) -> bool:
    words = line.split()
    return (len(line) < 90 and not DATE_RE.match(line) and not line.startswith(("RBI/", "DOR", "DBR", "DoR")) and not line.endswith((".", ";", ":", ",")) and len(words) <= 12
            and " | " not in line and not SUB_RE.match(line))


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.;])\s+(?=[A-Z(])|\n", text)
    parts, current = [], ""
    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > max_chars:
            parts.append(current.strip())
            current = ""
        current += " " + sentence
    if current.strip():
        parts.append(current.strip())
    return parts


def _drop_table_of_contents(paragraphs: list[str], min_run: int = 8) -> list[str]:
    """Remove a table of contents: a long run of short, unpunctuated lines near the top.

    Without this, TOC entries such as "Annex IV" or "2.1 Non-performing Assets" would be
    mistaken for the start of an annex or a numbered paragraph."""
    limit = max(1, len(paragraphs) // 4)
    i = 0
    while i < limit:
        j = i
        while j < len(paragraphs) and len(paragraphs[j]) < 200 and not paragraphs[j].endswith((".", ";", ":")):
            j += 1
        if j - i >= min_run:
            return paragraphs[:i] + paragraphs[j:]
        i = j + 1
    return paragraphs


def chunk_document(text: str, source: str) -> list[Chunk]:
    meta, body = split_header(text)
    code = doc_code(source)
    title, reference, url = meta.get("source", ""), meta.get("reference", ""), meta.get("url", "")

    # Paragraphs are separated by blank lines; drop site navigation and empty lines.
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body)]
    paragraphs = [p for p in paragraphs if p and not NAV_RE.match(p) and not DELETED_RE.match(p)]
    paragraphs = _drop_table_of_contents(paragraphs)

    blocks: list[dict] = []  # {"label", "section", "text", "heading"}
    num, sub, annex, section = "", "", "", ""

    def label() -> str:
        core = f"{num}({sub})" if num and sub else (num or (f"({sub})" if sub else "pre"))
        return f"{annex}/{core}" if annex else core

    for para in paragraphs:
        annex_m = ANNEX_RE.match(para)
        chapter_m = CHAPTER_RE.match(para)
        if annex_m and len(para) < 100:
            annex = "Ann" + (annex_m.group(2) or "")
            num, sub, section = "", "", para
            continue
        if chapter_m and len(para) < 140:
            section = para
            continue
        num_m, sub_m, fn_m = NUM_RE.match(para), SUB_RE.match(para), FOOTNOTE_RE.match(para)
        in_table = bool(blocks) and " | " in blocks[-1]["text"].splitlines()[-1]
        if in_table and len(para) < 150 and not para.endswith("."):
            blocks[-1]["text"] += "\n" + para  # table row such as "(a) Individual Housing Loans"
            continue
        if num_m:
            num, sub = num_m.group(1) or num_m.group(2), ""
            rest = para[num_m.end():]
            if _is_heading(rest):  # "3. QUANTUM OF LOAN" -> heading for what follows
                section = rest.title() if rest.isupper() else rest
                blocks.append({"label": label(), "section": section, "text": para, "heading": True})
                continue
        elif sub_m:
            sub = sub_m.group(1)
        elif fn_m and blocks and len(para) < 600 and re.search(r"circular|dated|vide|w\.e\.f|defined|substituted|inserted", para, re.I):
            blocks.append({"label": f"fn{fn_m.group(1)}", "section": "Footnotes", "text": para, "heading": False})
            continue
        elif blocks and (" | " in para or len(para) < MIN_CHARS or not _is_heading(para)):
            # continuation, table row or short line: attach to the current paragraph
            prev = blocks[-1]
            if len(prev["text"]) + len(para) < MAX_CHARS * 1.5 or " | " in para:
                prev["text"] += "\n" + para
                prev["heading"] = False
                continue
        elif _is_heading(para):
            section = para.title() if para.isupper() else para
            continue
        blocks.append({"label": label(), "section": section, "text": para, "heading": False})

    # A bare heading block absorbs the paragraph that follows it.
    merged: list[dict] = []
    for block in blocks:
        if merged and merged[-1]["heading"]:
            merged[-1]["text"] += "\n" + block["text"]
            merged[-1]["heading"] = False
            merged[-1]["label"] = block["label"]
            continue
        merged.append(block)

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for block in merged:
        for i, part in enumerate(_split_long(block["text"])):
            base = f"{code} ¶{block['label']}" + (f"-{i + 1}" if i else "")
            n = seen.get(base, 0)
            seen[base] = n + 1
            cid = base if n == 0 else f"{base}-{chr(ord('a') + n)}"
            chunks.append(Chunk(cid, part, source, block["section"], title, reference, url))
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
    embeddings = model.encode([c.indexed_text for c in chunks], normalize_embeddings=True, batch_size=64).tolist()

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    collection = client.create_collection(config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    for start in range(0, len(chunks), 500):
        batch = chunks[start:start + 500]
        collection.add(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embeddings[start:start + 500],
            metadatas=[{"source": c.source, "section": c.section, "reference": c.reference} for c in batch],
        )
    print(f"Indexed {len(chunks)} paragraphs from {policy_dir} into {chroma_dir}")
    return len(chunks)


if __name__ == "__main__":
    build_vector_store()
