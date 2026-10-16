# Lending Policy Retrieval & Eligibility Assistant

A Streamlit application for searching lending policy text and checking applicant eligibility against explicit rules. It uses local sentence-transformer embeddings and a persistent Chroma vector store to retrieve relevant clauses, alongside a separate rule-based eligibility checker.

The current application returns retrieved policy text directly. It does not call a generative language model to compose answers, and eligibility rules are defined in code rather than extracted from the documents.

## Features

- Load policy `.txt` files from `data/policies/`.
- Split documents into 350-character chunks with 40-character overlap.
- Embed chunks with `sentence-transformers/all-MiniLM-L6-v2` and store them in `data/chromadb/`.
- Retrieve the two most relevant chunks for a policy question.
- Check monthly income, CIBIL score, and DTI/FOIR for Personal, Housing, and Business loans.
- Includes a small standalone Ragas evaluation example.

## Quick start

Run commands from the repository root in a Python environment compatible with the dependencies.

```bash
git clone https://github.com/yash9518/lending-policy-rag.git
cd lending-policy-rag
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install packages, index the included policy, and start the application:

```bash
python -m pip install -r requirements.txt
python -m pip install langchain-text-splitters
python -m src.ingest
python -m streamlit run app.py
```

`langchain-text-splitters` is imported by the ingestion script but is not explicitly listed in `requirements.txt`, so it is installed separately above. The first embedding-model load requires internet access to download model files. The main application does not require an LLM API key.

Open the URL printed by Streamlit. Use **Policy Query (RAG)** to search clauses, or **Eligibility Verification Engine** to check an applicant.

## Eligibility logic implemented in code

| Check | Implemented threshold |
| --- | --- |
| Monthly income | At least INR 25,000 |
| CIBIL for Personal loans | At least 675 |
| CIBIL for Housing / Business loans | At least 650 |
| DTI/FOIR when income is below INR 50,000 | At most 50% |
| DTI/FOIR when income is at least INR 50,000 | At most 65% |

These thresholds describe this repository's implementation. The supplied policy text contains additional criteria, including age, self-employment income, loan limits, and overdue history, which the checker does not evaluate. Editing policy text does not change the rules in `src/agent.py`.

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit interface |
| `src/ingest.py` | Load, split, embed, and index policy text |
| `src/agent.py` | Retrieve clauses and evaluate eligibility |
| `data/policies/retail_lending_guidelines.txt` | Included policy document |
| `evaluation/eval_ragas.py` | Ragas example using three fixed question/context/answer pairs |
| `requirements.txt` | Python dependencies |

## Updating policies and evaluating

Add or edit `.txt` files in `data/policies/`. To rebuild an index, stop the app, remove the generated `data/chromadb/` directory, and rerun `python -m src.ingest`. Repeated ingestion into an existing store can retain or duplicate previous content. The generated index is ignored by Git.

The optional evaluation script can be invoked with:

```bash
python -m evaluation.eval_ragas
```

It evaluates hard-coded examples, not live retrieval from the application. Configure the evaluator's model provider and credentials as required by the installed Ragas version before running it; those settings are not supplied here. Dependency versions are unpinned, so library compatibility may require adjustment.
