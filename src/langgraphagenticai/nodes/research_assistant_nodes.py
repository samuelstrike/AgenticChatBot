from src.langgraphagenticai.state.state import State


RESEARCH_SYSTEM_PROMPT = (
    "You are an expert research assistant. When answering questions, use the available "
    "search tools to find accurate, up-to-date information. Always search before answering "
    "factual questions. Synthesize information from multiple sources into a clear, "
    "well-structured response with key findings highlighted."
)


class ResearchAssistantNode:
    def __init__(self, llm, tools: list):
        self.llm = llm
        self.tools = tools
        self._agent = None

    def _build_agent(self):
        from langchain_core.messages import SystemMessage
        llm_with_tools = self.llm.bind_tools(self.tools)

        def agent_node(state: State) -> State:
            messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        return agent_node

    def get_agent_node(self):
        return self._build_agent()
