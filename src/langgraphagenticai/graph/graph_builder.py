from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.state.agent_state import CodeAgentState, MultiAgentState
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.research_assistant_nodes import ResearchAssistantNode
from src.langgraphagenticai.nodes.code_assistant_nodes import CodeAssistantNodes
from src.langgraphagenticai.nodes.multiagent_nodes import MultiAgentNodes
from src.langgraphagenticai.tools.search_tools import get_search_tools


class GraphBuilder:
    def __init__(self, model, checkpointer: MemorySaver = None):
        self.llm = model
        self.checkpointer = checkpointer or MemorySaver()

    # ------------------------------------------------------------------
    # Use Case 1: Basic Chatbot
    # ------------------------------------------------------------------
    def basic_chatbot_build_graph(self):
        graph = StateGraph(State)
        node = BasicChatbotNode(self.llm)
        graph.add_node("chatbot", node.process)
        graph.add_edge(START, "chatbot")
        graph.add_edge("chatbot", END)
        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Use Case 2: AI Research Assistant (ReAct loop)
    # ------------------------------------------------------------------
    def research_assistant_build_graph(self, tavily_api_key: str = ""):
        tools = get_search_tools(tavily_api_key)
        research_node = ResearchAssistantNode(self.llm, tools)

        graph = StateGraph(State)
        graph.add_node("agent", research_node.get_agent_node())
        graph.add_node("tools", ToolNode(tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", tools_condition)
        graph.add_edge("tools", "agent")

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Use Case 3: Code Assistant (planner → coder → reviewer)
    # ------------------------------------------------------------------
    def code_assistant_build_graph(self):
        nodes = CodeAssistantNodes(self.llm)

        graph = StateGraph(CodeAgentState)
        graph.add_node("planner", nodes.planner_node)
        graph.add_node("coder", nodes.coder_node)
        graph.add_node("reviewer", nodes.reviewer_node)

        graph.add_edge(START, "planner")
        graph.add_edge("planner", "coder")
        graph.add_edge("coder", "reviewer")
        graph.add_edge("reviewer", END)

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Use Case 4: Multi-Agent Research Team (supervisor pattern)
    # ------------------------------------------------------------------
    def multiagent_build_graph(self, tavily_api_key: str = ""):
        tools = get_search_tools(tavily_api_key)
        agents = MultiAgentNodes(self.llm, tools)

        graph = StateGraph(MultiAgentState)
        graph.add_node("supervisor", agents.supervisor_node)
        graph.add_node("researcher", agents.researcher_node)
        graph.add_node("analyst", agents.analyst_node)
        graph.add_node("writer", agents.writer_node)

        graph.add_edge(START, "supervisor")

        def route_from_supervisor(state: MultiAgentState) -> str:
            next_agent = state.get("next_agent", "FINISH")
            return next_agent if next_agent != "FINISH" else END

        graph.add_conditional_edges(
            "supervisor",
            route_from_supervisor,
            {"researcher": "researcher", "analyst": "analyst", "writer": "writer", END: END},
        )
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")

        return graph.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------
    def setup_graph(self, usecase: str, tavily_api_key: str = ""):
        if usecase == "Basic Chatbot":
            return self.basic_chatbot_build_graph()
        elif usecase == "AI Research Assistant":
            return self.research_assistant_build_graph(tavily_api_key)
        elif usecase == "Code Assistant":
            return self.code_assistant_build_graph()
        elif usecase == "Multi-Agent Research Team":
            return self.multiagent_build_graph(tavily_api_key)
        else:
            raise ValueError(f"Unknown use case: {usecase}")
