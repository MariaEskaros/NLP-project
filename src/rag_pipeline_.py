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
        max_history_chars: int = 1500,
        semantic_cache=None,
        use_cache: bool = False,
        out_of_domain_threshold: float = 0.45,
        fallback_llm=None,
        max_retries: int = 2
    ):
        self.retriever = retriever
        self.llm = llm
        self.prompt_type = prompt_type

        self.memory = memory
        self.use_memory = use_memory
        self.memory_strategy = memory_strategy
        self.last_n_turns = last_n_turns
        self.max_history_chars = max_history_chars

        self.semantic_cache = semantic_cache
        self.use_cache = use_cache

        self.out_of_domain_threshold = out_of_domain_threshold
        self.fallback_llm = fallback_llm
        self.max_retries = max_retries

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

    def is_out_of_domain(self, retrieved_chunks):
        if not retrieved_chunks:
            return True, None

        best_score = retrieved_chunks[0].get("relevance_score")

        if best_score is None:
            return False, None

        if best_score < self.out_of_domain_threshold:
            return True, best_score

        return False, best_score

    def generate_with_retry_and_fallback(self, prompt: str):
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self.llm.generate(prompt), getattr(self.llm, "model_name", "unknown"), None
            except Exception as e:
                last_error = e

        if self.fallback_llm is not None:
            try:
                return self.fallback_llm.generate(prompt), getattr(self.fallback_llm, "model_name", "fallback"), str(last_error)
            except Exception as fallback_error:
                return (
                    "حدث خطأ أثناء توليد الإجابة. من فضلك حاول مرة أخرى لاحقا.",
                    "fallback_failed",
                    str(fallback_error)
                )

        return (
            "حدث خطأ أثناء توليد الإجابة. من فضلك حاول مرة أخرى لاحقا.",
            "primary_failed",
            str(last_error)
        )

    def answer(self, question: str):

        # 1. Check semantic cache
        if self.use_cache and self.semantic_cache is not None:
            cache_result = self.semantic_cache.lookup(question)

            if cache_result and cache_result.get("cache_hit"):
                cached_answer = cache_result["answer"]

                if self.use_memory and self.memory is not None:
                    self.memory.add_user_message(question)
                    self.memory.add_ai_message(cached_answer)

                return {
                    "question": question,
                    "answer": cached_answer,
                    "retrieved_chunks": [],
                    "prompt_type": self.prompt_type,
                    "model": cache_result.get("model", "unknown"),
                    "used_memory": self.use_memory,
                    "memory_strategy": self.memory_strategy,
                    "prompt_chars": 0,
                    "cache_hit": True,
                    "cache_similarity": cache_result.get("similarity"),
                    "matched_question": cache_result.get("matched_question"),
                    "out_of_domain": False,
                    "error": None
                }

        # 2. Retrieve chunks
        retrieved_chunks = self.retriever.retrieve(question)

        # 3. Out-of-domain detection
        out_of_domain, best_score = self.is_out_of_domain(retrieved_chunks)

        if out_of_domain:
            answer = "السؤال خارج نطاق المحتوى المتاح."

            return {
                "question": question,
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "prompt_type": self.prompt_type,
                "model": None,
                "used_memory": self.use_memory,
                "memory_strategy": self.memory_strategy,
                "prompt_chars": 0,
                "cache_hit": False,
                "cache_similarity": None,
                "matched_question": None,
                "out_of_domain": True,
                "retrieval_score": best_score,
                "error": None
            }

        # 4. Build prompt
        prompt = self.build_prompt(
            question=question,
            retrieved_chunks=retrieved_chunks
        )

        # 5. Generate safely
        answer, model_used, error = self.generate_with_retry_and_fallback(prompt)

        # 6. Save answer in cache
        if self.use_cache and self.semantic_cache is not None:
            self.semantic_cache.add(
                question=question,
                answer=answer,
                model=model_used
            )

        # 7. Update memory
        if self.use_memory and self.memory is not None:
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "prompt_type": self.prompt_type,
            "model": model_used,
            "used_memory": self.use_memory,
            "memory_strategy": self.memory_strategy,
            "prompt_chars": len(prompt),
            "cache_hit": False,
            "cache_similarity": None,
            "matched_question": None,
            "out_of_domain": False,
            "retrieval_score": best_score,
            "error": error
        }