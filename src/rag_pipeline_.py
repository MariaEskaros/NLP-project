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
        use_memory: bool = True,
        memory_strategy: str = "full",
        last_n_turns: int = 2,
        max_history_chars: int = 1500
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_type = prompt_type
        self.memory = memory
        self.use_memory = use_memory
        self.memory_strategy = memory_strategy
        self.last_n_turns = last_n_turns
        self.max_history_chars = max_history_chars

    def build_prompt(self, question: str, retrieved_chunks):
        if self.prompt_type == "minimal":
            base_prompt = build_minimal_prompt(question, retrieved_chunks)

        elif self.prompt_type == "arabic":
            base_prompt = build_arabic_strict_prompt(question, retrieved_chunks)

        else:
            base_prompt = build_strict_grounded_prompt(question, retrieved_chunks)

        if self.use_memory and self.memory is not None:
            history_text = self.memory.get_history_text(
                strategy=self.memory_strategy,
                last_n_turns=self.last_n_turns,
                max_chars=self.max_history_chars,
                llm=self.llm
            )

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
            "used_memory": self.use_memory,
            "memory_strategy": self.memory_strategy,
            "prompt_chars": len(prompt)
        }