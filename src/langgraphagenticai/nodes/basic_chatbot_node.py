from src.langgraphagenticai.state.state import State

class BasicChatbotNode:
    """
    A basic chatbot node that can be used in a LangGraph graph.
    This node is designed to handle user input and provide responses.
    """

    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> State:
        """
        Processes the inout state and generates a chat response.
        """

        return {"messages": self.llm.invoke(state["messages"])}