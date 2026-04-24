from typing import Literal
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.langgraphagenticai.state.agent_state import MultiAgentState


class RouteDecision(BaseModel):
    next: Literal["researcher", "analyst", "writer", "FINISH"]
    reasoning: str


SUPERVISOR_SYSTEM = """You are a research team supervisor managing three specialists:
- researcher: Searches the web and gathers raw information on a topic
- analyst: Analyses gathered data and identifies key insights, trends, and patterns
- writer: Writes a polished, well-structured final report from the analysis

Given the conversation history and work done so far, decide which specialist should act next.
Respond with FINISH only when the writer has produced a final report.
Always start with researcher, then analyst, then writer unless work is already done."""

RESEARCHER_SYSTEM = (
    "You are a research specialist. Use your search tools to gather comprehensive, "
    "accurate information on the topic. Summarise your findings clearly with source context."
)

ANALYST_SYSTEM = (
    "You are a data analyst. Review the research findings and produce a structured analysis: "
    "key trends, important data points, implications, and any gaps or caveats in the research."
)

WRITER_SYSTEM = (
    "You are a professional report writer. Using the research and analysis provided, write a "
    "comprehensive, well-structured final report. Use headings, bullet points where appropriate, "
    "and end with a concise executive summary."
)


class MultiAgentNodes:
    def __init__(self, llm, tools: list):
        self.llm = llm
        self.tools = tools
        self.llm_with_tools = llm.bind_tools(tools) if tools else llm

    def supervisor_node(self, state: MultiAgentState) -> MultiAgentState:
        messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + list(state["messages"])
        try:
            structured_llm = self.llm.with_structured_output(RouteDecision)
            decision: RouteDecision = structured_llm.invoke(messages)
            next_agent = decision.next
            note = f"[Supervisor] → {next_agent}: {decision.reasoning}"
        except Exception:
            # Fallback routing based on agent_notes length
            notes = state.get("agent_notes", [])
            if len(notes) == 0:
                next_agent = "researcher"
            elif len(notes) == 1:
                next_agent = "analyst"
            elif len(notes) == 2:
                next_agent = "writer"
            else:
                next_agent = "FINISH"
            note = f"[Supervisor] → {next_agent} (fallback routing)"

        return {"next_agent": next_agent, "agent_notes": [note]}

    def researcher_node(self, state: MultiAgentState) -> MultiAgentState:
        user_query = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break

        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM),
            HumanMessage(content=f"Research this topic thoroughly: {user_query}"),
        ]
        response = self.llm_with_tools.invoke(messages)

        # If tool calls exist, execute them manually
        if hasattr(response, "tool_calls") and response.tool_calls:
            from langchain_core.messages import ToolMessage
            tool_results = []
            for tc in response.tool_calls:
                for tool in self.tools:
                    if tool.name == tc["name"]:
                        try:
                            result = tool.invoke(tc["args"])
                            tool_results.append(
                                ToolMessage(content=str(result), tool_call_id=tc["id"])
                            )
                        except Exception as e:
                            tool_results.append(
                                ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"])
                            )

            # Final synthesis after tool results
            synthesis_messages = messages + [response] + tool_results + [
                HumanMessage(content="Now synthesise the above search results into a clear research summary.")
            ]
            final_response = self.llm.invoke(synthesis_messages)
            note = "[Researcher] Completed web research and synthesis."
            return {"messages": [final_response], "agent_notes": [note]}

        note = "[Researcher] Completed research (no external search needed)."
        return {"messages": [response], "agent_notes": [note]}

    def analyst_node(self, state: MultiAgentState) -> MultiAgentState:
        messages = [SystemMessage(content=ANALYST_SYSTEM)] + list(state["messages"]) + [
            HumanMessage(content="Analyse the research above and provide structured insights.")
        ]
        response = self.llm.invoke(messages)
        note = "[Analyst] Completed analysis and identified key insights."
        return {"messages": [response], "agent_notes": [note]}

    def writer_node(self, state: MultiAgentState) -> MultiAgentState:
        messages = [SystemMessage(content=WRITER_SYSTEM)] + list(state["messages"]) + [
            HumanMessage(content="Write the final comprehensive report based on the research and analysis above.")
        ]
        response = self.llm.invoke(messages)
        note = "[Writer] Final report produced."
        return {"messages": [response], "agent_notes": [note]}
