from prompts import (
    build_minimal_prompt,
    build_strict_grounded_prompt,
    build_arabic_strict_prompt
)


class RAGPipeline:
    def __init__(
        self,
        retriever,
        llm,
        prompt_type: str = "strict",
        memory=None,
        use_memory: bool = True
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_type = prompt_type
        self.memory = memory
        self.use_memory = use_memory

    def build_prompt(self, question: str, retrieved_chunks):
        if self.prompt_type == "minimal":
            base_prompt = build_minimal_prompt(question, retrieved_chunks)

        elif self.prompt_type == "arabic":
            base_prompt = build_arabic_strict_prompt(question, retrieved_chunks)

        else:
            base_prompt = build_strict_grounded_prompt(question, retrieved_chunks)

        if self.use_memory and self.memory is not None:
            history_text = self.memory.get_full_history_text()

            if history_text.strip():
                base_prompt = f"""
Conversation history:
{history_text}

Current turn:
{base_prompt}
""".strip()

        return base_prompt

    def answer(self, question: str):
        retrieved_chunks = self.retriever.retrieve(question)

        prompt = self.build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        answer = self.llm.generate(prompt)

        if self.use_memory and self.memory is not None:
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "prompt_type": self.prompt_type,
            "model": getattr(self.llm, "model_name", "unknown"),
            "used_memory": self.use_memory
        }