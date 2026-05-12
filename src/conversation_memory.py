from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:
    def __init__(self):
        self.messages = []
        self.summary = ""

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def get_full_history_text(self) -> str:
        return self._messages_to_text(self.messages)

    def get_sliding_window_text(self, last_n_turns: int = 2) -> str:
        # each turn = user + assistant, so 2 messages per turn
        selected_messages = self.messages[-last_n_turns * 2:]
        return self._messages_to_text(selected_messages)

    def get_truncated_history_text(self, max_chars: int = 1500) -> str:
        full_text = self.get_full_history_text()

        if len(full_text) <= max_chars:
            return full_text

        return full_text[-max_chars:]
    
    def get_summarized_history_text(self, llm) -> str:
        full_text = self.get_full_history_text()

        if not full_text.strip():
            return ""

        summary_prompt = f"""
    Summarize the following conversation briefly.
    Keep only information useful for answering future follow-up questions.
    Preserve important Arabic terms and episode references.

    Conversation:
    {full_text}

    Summary:
    """.strip()

        self.summary = llm.generate(summary_prompt)
        return self.summary

    def get_history_text(
        self,
        strategy: str = "full",
        last_n_turns: int = 2,
        max_chars: int = 1500,
        llm=None
    ) -> str:
        if strategy == "sliding":
            return self.get_sliding_window_text(last_n_turns=last_n_turns)

        if strategy == "truncated":
            return self.get_truncated_history_text(max_chars=max_chars)

        if strategy == "summarized":
            if llm is None:
                return self.get_full_history_text()
            return self.get_summarized_history_text(llm)

        return self.get_full_history_text()

    def _messages_to_text(self, messages) -> str:
        history = []

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                history.append(f"Assistant: {message.content}")

        return "\n".join(history)

    def clear(self):
        self.messages = []