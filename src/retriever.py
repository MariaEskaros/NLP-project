class Retriever:
    def __init__(self, embedding_model, vector_store, top_k: int = 5):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str):
        query_embedding = self.embedding_model.embed_query(query)

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.top_k
        )

        retrieved_chunks = []

        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]

            # Chroma distance: lower is better
            relevance_score = 1 / (1 + distance)

            retrieved_chunks.append({
                "rank": i + 1,
                "text": results["documents"][0][i],
                "episode": results["metadatas"][0][i]["episode"],
                "source_file": results["metadatas"][0][i]["source_file"],
                "distance": distance,
                "relevance_score": relevance_score
            })

        return retrieved_chunks