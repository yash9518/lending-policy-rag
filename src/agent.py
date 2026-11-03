import os

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.llm import build_llm, message_text

load_dotenv()

CHROMA_PATH = "data/chromadb"

ANSWER_SYSTEM_PROMPT = (
    "You are a retail lending policy assistant. Answer the user's question using only "
    "the policy excerpts provided. Quote figures and conditions exactly as written. "
    "If the excerpts do not contain the answer, say that the policy text does not cover it. "
    "Do not make credit decisions and do not infer applicant eligibility. "
    "Keep the answer to a few sentences."
)


class PolicyAgent:
    def __init__(self):
        if not os.path.isdir(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
            # Cloud hosts start from a fresh checkout without the git-ignored index.
            from src.ingest import build_vector_store
            build_vector_store()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        self.llm, self.llm_status = build_llm()

    def retrieve_policy(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return "No matching policy clause located."
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    def answer_policy_question(self, query: str) -> dict:
        """Retrieve clauses and, if an LLM is configured, write a grounded answer.

        Returns a dict with:
          answer:  generated answer text, or None if no LLM is available
          sources: list of retrieved clause texts
          message: explanation when no answer was generated
        """
        docs = self.retriever.invoke(query)
        sources = [doc.page_content for doc in docs]
        if not sources:
            return {"answer": None, "sources": [], "message": "No matching policy clause located."}

        if self.llm is None:
            return {"answer": None, "sources": sources, "message": self.llm_status}

        context = "\n\n---\n\n".join(sources)
        response = self.llm.invoke([
            ("system", ANSWER_SYSTEM_PROMPT),
            ("human", f"Policy excerpts:\n\n{context}\n\nQuestion: {query}"),
        ])
        return {"answer": message_text(response), "sources": sources, "message": None}

    def calculate_eligibility(self, income: float, dti: float, cibil: int, loan_type: str = "Personal") -> dict:
        min_cibil = 675 if loan_type.lower() == "personal" else 650
        min_income = 25000.0
        max_dti = 0.50 if income < 50000 else 0.65

        reasons = []
        is_eligible = True

        if cibil < min_cibil:
            is_eligible = False
            reasons.append(f"CIBIL score {cibil} is below the required {min_cibil} minimum.")
        if income < min_income:
            is_eligible = False
            reasons.append(f"Monthly income INR {income:,.0f} does not satisfy INR {min_income:,.0f} baseline.")
        if dti > max_dti:
            is_eligible = False
            reasons.append(f"DTI/FOIR of {dti:.1%} exceeds the permitted {max_dti:.1%} cap.")

        return {
            "eligible": is_eligible,
            "reasons": reasons if not is_eligible else ["Application strictly complies with policy parameters."]
        }
