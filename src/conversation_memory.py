from langchain_core.messages import HumanMessage, AIMessage


class ConversationMemory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def get_full_history_text(self) -> str:
        history = []

        for message in self.messages:
            if isinstance(message, HumanMessage):
                history.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                history.append(f"Assistant: {message.content}")

        return "\n".join(history)

    def clear(self):
        self.messages = []