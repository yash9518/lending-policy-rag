from src.ingest import chunk_document


def test_chunker_keeps_one_clause_per_bullet():
    text = "TITLE\n\n1. FIRST SECTION:\n- Rule A: 10%.\n- Rule B: 20%.\n\n2. SECOND:\n- Rule C.\n"
    chunks = chunk_document(text, "doc.txt")
    assert [c.id for c in chunks] == ["doc §0.1", "doc §1.1", "doc §1.2", "doc §2.1"]
    assert chunks[1].section == "First Section" and chunks[1].text == "Rule A: 10%."


def test_chunker_handles_unstructured_text():
    chunks = chunk_document("Plain paragraph one.\n\nPlain paragraph two.", "notes.txt")
    assert [c.text for c in chunks] == ["Plain paragraph one.", "Plain paragraph two."]


def test_bm25_finds_expected_clauses(retriever):
    cases = {
        "maximum LTV for affordable housing loans": "§2.2",
        "minimum CIBIL score for personal loans": "§3.1",
        "how many hard credit inquiries are allowed": "§3.2",
        "prohibited industries": "§3.4",
        "minimum income for self-employed": "§1.2",
    }
    for query, expected in cases.items():
        top = [c.chunk.id for c in retriever.search(query, k=2)]
        assert any(i.endswith(expected) for i in top), (query, top)
