import pytest

from src.ingest import chunk_document, doc_code

SAMPLE = """SOURCE: Test Circular
REFERENCE: RBI/2099-00/1
URL: https://example.invalid
=====

RBI/2099-00/1

Madam / Dear Sir,

Contents

1 | Introduction

2 | Rules

3 | Other

Annex A

Annex B

Appendix

List of circulars

Something else

2. The regulated entities shall do the first thing in every case without exception.

3. On review, the following instructions are issued.

(i) Penalty shall be a charge and not interest.

(ii) Charges shall be disclosed.

4. QUANTUM OF LOAN

(a) Banks should abide by the following limits:

Category | LTV (%)

(a) Small loans

Upto 30 lakh | 90

Annex I

1. Scope: These guidelines apply to all lenders.

1 As defined in the circular dated January 4, 2018.

2026

All Months
"""


def test_doc_codes():
    assert doc_code("PEN.txt") == "PEN"
    assert doc_code("retail_lending_guidelines.txt") == "RLG"


def test_chunker_follows_rbi_paragraph_numbering():
    chunks = chunk_document(SAMPLE, "TST.txt")
    by = {c.id: c for c in chunks}
    assert {"TST ¶2", "TST ¶3", "TST ¶3(i)", "TST ¶3(ii)", "TST ¶4(a)", "TST ¶AnnI/1"} <= set(by)
    assert by["TST ¶3(i)"].text.startswith("(i) Penalty")
    assert by["TST ¶3(i)"].reference == "RBI/2099-00/1"
    # table rows (including "(a) Small loans") stay with the paragraph that introduces them
    assert "Upto 30 lakh | 90" in by["TST ¶4(a)"].text and "(a) Small loans" in by["TST ¶4(a)"].text
    assert by["TST ¶4(a)"].section == "Quantum Of Loan"
    # table of contents and site navigation are dropped, not mistaken for an annex
    assert not any("Annex A" in c.text or "All Months" in c.text for c in chunks)
    assert not any(c.id.startswith("TST ¶AnnA") for c in chunks)


def test_corpus_is_official_rbi_text(retriever):
    chunks = retriever.chunks
    assert len(chunks) > 800 and len({c.id for c in chunks}) == len(chunks)
    assert {c.source for c in chunks} == {f"{d}.txt" for d in ("DLG", "DOC", "EMI", "HFC", "KFS", "KYC", "NPA", "PEN")}
    assert all(c.reference.startswith("RBI/") and c.url.startswith("https://www.rbi.org.in/") for c in chunks)


@pytest.mark.parametrize("query, expected", [
    ("loan to value ratio limits for individual housing loans", "HFC ¶3(a)"),
    ("penal charges must not be penal interest added to the rate", "PEN ¶3(i)"),
    ("compensation per day of delay in releasing property documents", "DOC ¶6"),
    ("validity period of the key facts statement", "KFS ¶5"),
    ("cooling off period for digital loans", "DLG ¶AnnI/8"),
    ("special mention account SMA-1 SMA-2 classification", "NPA ¶8.1"),
])
def test_bm25_finds_expected_paragraphs(retriever, query, expected):
    top = [c.chunk.id for c in retriever.search(query, k=5)]
    assert expected in top, (query, top)
