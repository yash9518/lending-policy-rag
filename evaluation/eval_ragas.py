from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def test_evals():
    eval_dataset = Dataset.from_dict({
        "question": [
            "What is the minimum age to apply for a retail loan?",
            "What is the minimum CIBIL score for unsecured personal loans?",
            "What is the maximum LTV for affordable housing loans under 30 Lakhs?"
        ],
        "contexts": [
            ["Minimum Age: 21 years at loan application; Maximum Age: 60 years at loan maturity."],
            ["Minimum Bureau Score (CIBIL): 675 for unsecured personal loans; 650 for secured housing/business loans."],
            ["Affordable Housing Loans: Up to INR 35,00,000. Maximum LTV: 80% for loans up to 30 Lakhs, 75% above 30 Lakhs."]
        ],
        "answer": [
            "The minimum age requirement is 21 years.",
            "The minimum CIBIL score required is 675.",
            "The maximum loan-to-value (LTV) is 80% for loans up to 30 Lakhs."
        ]
    })

    print("Benchmarking RAG pipeline with Ragas evaluation metrics...")
    results = evaluate(eval_dataset, metrics=[faithfulness, answer_relevancy])
    print(results)

if __name__ == "__main__":
    test_evals()