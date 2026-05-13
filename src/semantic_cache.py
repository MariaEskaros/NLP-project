import json
from pathlib import Path
import numpy as np


class SemanticCache:
    def __init__(
        self,
        embedding_model,
        cache_path: Path,
        similarity_threshold: float = 0.88
    ):
        self.embedding_model = embedding_model
        self.cache_path = Path(cache_path)
        self.similarity_threshold = similarity_threshold
        self.entries = []

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        else:
            self.entries = []

    def save(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def lookup(self, question: str):
        if not self.entries:
            return {
                "cache_hit": False,
                "similarity": None,
                "matched_question": None
            }

        query_embedding = self.embedding_model.embed_query(question)

        best_entry = None
        best_score = -1

        for entry in self.entries:
            score = self.cosine_similarity(
                query_embedding,
                entry["question_embedding"]
            )

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.similarity_threshold:
            return {
                "cache_hit": True,
                "similarity": best_score,
                "matched_question": best_entry["question"],
                "answer": best_entry["answer"],
                "model": best_entry.get("model", "unknown")
            }

        return {
            "cache_hit": False,
            "similarity": best_score,
            "matched_question": best_entry["question"] if best_entry else None
        }

    def add(self, question: str, answer: str, model: str = "unknown"):
        question_embedding = self.embedding_model.embed_query(question)

        self.entries.append({
            "question": question,
            "answer": answer,
            "model": model,
            "question_embedding": question_embedding.tolist()
        })

        self.save()

    def clear(self):
        self.entries = []
        self.save()

    def count(self):
        return len(self.entries)