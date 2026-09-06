from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PATH = "data/chromadb"

class PolicyAgent:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

    def retrieve_policy(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        if not docs:
            return "No matching policy clause located."
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

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