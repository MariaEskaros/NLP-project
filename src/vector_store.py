from pathlib import Path
from typing import List, Dict
import chromadb


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = "arabic_transcripts"):
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(path=self.persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Arabic transcript chunks for MS3 RAG"}
        )

    def add_chunks(self, chunks: List[Dict], embeddings):
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["text"] for chunk in chunks]

        metadatas = [
            {
                "episode": chunk["episode"],
                "source_file": chunk["source_file"]
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k: int = 5):
        # NEW: Check if it's a NumPy array (has .tolist()) or already a standard list
        if hasattr(query_embedding, "tolist"):
            embedding_list = query_embedding.tolist()
        else:
            embedding_list = query_embedding

        results = self.collection.query(
            query_embeddings=[embedding_list],
            n_results=top_k
        )

        return results

    def count(self):
        return self.collection.count()