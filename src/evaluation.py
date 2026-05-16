import json
import re
import time
from pathlib import Path

import numpy as np


def load_test_sets(test_dir: Path):
    """
    Load all *_test_set.json files from data/QA test.
    Supports SQuAD-like format:
    data -> paragraphs -> qas -> answers
    """

    test_items = []

    for file_path in sorted(test_dir.glob("*_test_set.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # SQuAD-like format
        if isinstance(data, dict) and "data" in data:
            for article in data["data"]:
                title = article.get("title", "")

                for paragraph in article.get("paragraphs", []):
                    context = paragraph.get("context", "")

                    for qa in paragraph.get("qas", []):
                        question = qa.get("question", "")

                        answers = qa.get("answers", [])
                        answer_text = ""

                        if answers:
                            first_answer = answers[0]

                            if isinstance(first_answer, dict):
                                answer_text = first_answer.get("text", "")
                            else:
                                answer_text = str(first_answer)

                        if question and answer_text:
                            test_items.append({
                                "question": question,
                                "reference_answer": answer_text,
                                "context": context,
                                "title": title,
                                "source_file": file_path.name,
                                "id": qa.get("id", "")
                            })

        # Flat list format
        elif isinstance(data, list):
            for item in data:
                question = (
                    item.get("question")
                    or item.get("query")
                    or item.get("input")
                )

                answer = (
                    item.get("answer")
                    or item.get("reference_answer")
                    or item.get("ground_truth")
                    or item.get("expected_answer")
                    or item.get("answers")
                )

                if isinstance(answer, list):
                    if len(answer) > 0 and isinstance(answer[0], dict):
                        answer = answer[0].get("text", "")
                    elif len(answer) > 0:
                        answer = answer[0]
                    else:
                        answer = ""

                if question and answer:
                    test_items.append({
                        "question": question,
                        "reference_answer": answer,
                        "context": item.get("context", ""),
                        "title": item.get("title", ""),
                        "source_file": file_path.name,
                        "id": item.get("id", "")
                    })

    return test_items


def tokenize_text(text: str):
    return re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", str(text).lower())


def cosine_similarity(vec1, vec2):
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    if denominator == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / denominator)


def semantic_similarity_score(generated_answer, reference_answer, embedding_model):
    gen_emb = embedding_model.embed_query(generated_answer)
    ref_emb = embedding_model.embed_query(reference_answer)

    return cosine_similarity(gen_emb, ref_emb)


def rouge_l_score(generated_answer, reference_answer):
    """
    ROUGE-L F1 based on longest common subsequence.
    """

    gen_tokens = tokenize_text(generated_answer)
    ref_tokens = tokenize_text(reference_answer)

    if not gen_tokens or not ref_tokens:
        return 0.0

    m = len(ref_tokens)
    n = len(gen_tokens)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == gen_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(
                    dp[i][j + 1],
                    dp[i + 1][j]
                )

    lcs = dp[m][n]

    precision = lcs / n
    recall = lcs / m

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def grounding_score(answer, retrieved_chunks):
    """
    Grounding score:
    percentage of generated answer tokens found in retrieved context.
    """

    if not retrieved_chunks:
        return 0.0

    context = " ".join(
        chunk.get("text", "")
        for chunk in retrieved_chunks
    )

    answer_tokens = set(tokenize_text(answer))
    context_tokens = set(tokenize_text(context))

    if not answer_tokens:
        return 0.0

    overlap = answer_tokens.intersection(context_tokens)

    return len(overlap) / len(answer_tokens)


def run_rag_evaluation(rag_pipeline, test_items, embedding_model):
    results = []

    for idx, item in enumerate(test_items, start=1):
        question = item["question"]
        reference_answer = item["reference_answer"]

        start_time = time.time()

        rag_result = rag_pipeline.answer(question)

        latency = time.time() - start_time

        generated_answer = rag_result.get("answer", "")
        retrieved_chunks = rag_result.get("retrieved_chunks", [])

        rouge_l = rouge_l_score(
            generated_answer,
            reference_answer
        )

        semantic_score = semantic_similarity_score(
            generated_answer,
            reference_answer,
            embedding_model
        )

        ground_score = grounding_score(
            generated_answer,
            retrieved_chunks
        )

        results.append({
            "question_id": idx,
            "source_file": item["source_file"],
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "rouge_l": rouge_l,
            "semantic_similarity": semantic_score,
            "grounding_score": ground_score,
            "latency_seconds": latency,
            "cache_hit": rag_result.get("cache_hit", False),
            "out_of_domain": rag_result.get("out_of_domain", False),
            "model": rag_result.get("model", "unknown"),
            "memory_strategy": rag_result.get("memory_strategy", "none")
        })

    return results