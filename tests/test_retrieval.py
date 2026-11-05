import pytest

from src.ingest import chunk_document, doc_code


def test_doc_codes():
    assert doc_code("retail_lending_guidelines.txt") == "RLG"
    assert doc_code("credit_deviations_and_authority.txt") == "CDA"


def test_chunker_keeps_one_clause_per_bullet():
    text = "TITLE\n\n1. FIRST SECTION:\n- Rule A: 10%.\n- Rule B: 20%.\n\n2. SECOND:\n- Rule C.\n"
    chunks = chunk_document(text, "doc.txt")
    assert [c.id for c in chunks] == ["D §0.1", "D §1.1", "D §1.2", "D §2.1"]
    assert chunks[1].section == "First Section" and chunks[1].text == "Rule A: 10%."


def test_chunker_handles_unstructured_text():
    chunks = chunk_document("Plain paragraph one.\n\nPlain paragraph two.", "notes.txt")
    assert [c.text for c in chunks] == ["Plain paragraph one.", "Plain paragraph two."]


def test_corpus_ids_unique(retriever):
    ids = [c.id for c in retriever.chunks]
    assert len(ids) == len(set(ids)) and len(ids) > 60


@pytest.mark.parametrize("query, expected", [
    ("maximum LTV for affordable housing loans", "RLG §2.2"),
    ("minimum CIBIL score for personal loans", "RLG §3.1"),
    ("how many hard credit inquiries are allowed", "RLG §3.2"),
    ("prohibited industries", "RLG §3.4"),
    ("minimum income for self-employed", "RLG §1.2"),
    ("foreclosure charges on floating rate loans", "PC §4.1"),
    ("when must property documents be released after closure", "FPG §3.1"),
    ("SMA-2 classification", "CR §1.3"),
    ("who can approve a bureau score deviation", "CDA §2.1"),
    ("re-KYC periodicity for high risk customers", "KD §4.2"),
])
def test_bm25_finds_expected_clauses(retriever, query, expected):
    top = [c.chunk.id for c in retriever.search(query, k=3)]
    assert expected in top, (query, top)
