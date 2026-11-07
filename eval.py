import re
from main_test import search
from llm import ask_llm

# Each question can have multiple acceptable/relevant functions
TEST_SET = [
    {
        "question": "how are models loaded for evaluation",
        "relevant_functions": ["load_models_and_data"],
    },
    {
        "question": "what does load_all do",
        "relevant_functions": ["load_all"],
    },
    {
        "question": "how is the model trained",
        "relevant_functions": ["train_model", "build_cnn_bilstm", "build_transformer_model"],
    },
    {
        "question": "how is model accuracy evaluated",
        "relevant_functions": ["evaluate_model"],
    },
    {
        "question": "how is data preprocessed for training",
        "relevant_functions": ["load_data", "clean_data"],
    },
]


def precision_recall_f1(retrieved, relevant):
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    true_positives = len(retrieved_set & relevant_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def check_hallucination(answer, retrieved_chunks):
    """Flags identifiers in the answer that don't appear anywhere in the retrieved code."""
    all_code_text = " ".join(c.get("code", "") for c in retrieved_chunks)
    mentioned = set(re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", answer))
    suspicious = [m for m in mentioned if m not in all_code_text and "_" in m]
    return suspicious


def evaluate(top_k=5, check_answers=False):
    hits = 0
    top1_correct = 0
    reciprocal_ranks = []
    precisions, recalls, f1s = [], [], []
    hallucination_flags = 0

    print(f"Running evaluation on {len(TEST_SET)} test questions (top_k={top_k})\n")

    for case in TEST_SET:
        question = case["question"]
        relevant = case["relevant_functions"]

        results = search(question, top_k=top_k)
        retrieved_functions = [r["function"] for r in results]

        found_relevant = [f for f in retrieved_functions if f in relevant]
        if found_relevant:
            hits += 1
            rank = retrieved_functions.index(found_relevant[0]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
            rank = None

        if retrieved_functions and retrieved_functions[0] in relevant:
            top1_correct += 1

        p, r, f1 = precision_recall_f1(retrieved_functions, relevant)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

        print(f"Q: {question}")
        print(f"   relevant: {relevant}")
        print(f"   retrieved: {retrieved_functions}")
        print(f"   rank of first match: {rank}  |  P: {p:.2f}  R: {r:.2f}  F1: {f1:.2f}")

        if check_answers:
            answer = ask_llm(question, results)
            suspicious = check_hallucination(answer, results)
            if suspicious:
                hallucination_flags += 1
                print(f"   ⚠️  possibly hallucinated references: {suspicious}")
            else:
                print(f"   ✅ answer grounded in retrieved code")

        print()

    n = len(TEST_SET)
    print("--- Summary ---")
    print(f"Hit Rate @ {top_k}: {hits/n*100:.1f}% ({hits}/{n})")
    print(f"Top-1 Accuracy: {top1_correct/n*100:.1f}% ({top1_correct}/{n})")
    print(f"Mean Reciprocal Rank: {sum(reciprocal_ranks)/n:.3f}")
    print(f"Precision @ {top_k}: {sum(precisions)/n:.3f}")
    print(f"Recall @ {top_k}: {sum(recalls)/n:.3f}")
    print(f"F1 @ {top_k}: {sum(f1s)/n:.3f}")
    if check_answers:
        print(f"Answers with possible hallucination: {hallucination_flags}/{n}")


if __name__ == "__main__":
    evaluate(top_k=5, check_answers=True)