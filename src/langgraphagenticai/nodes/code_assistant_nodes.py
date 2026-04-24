from langchain_core.messages import SystemMessage, HumanMessage
from src.langgraphagenticai.state.agent_state import CodeAgentState


PLANNER_SYSTEM = (
    "You are a software architect. Given the user's request, produce a concise numbered "
    "implementation plan (steps only, no code yet). Be specific about data structures, "
    "algorithms, and edge cases to handle."
)

CODER_SYSTEM = (
    "You are a senior software engineer. Implement exactly the plan provided. "
    "Write clean, well-structured code with brief inline comments only where non-obvious. "
    "Include a usage example at the end."
)

REVIEWER_SYSTEM = (
    "You are an expert code reviewer. Review the code for bugs, security issues, and "
    "improvements. List issues found, then provide the final corrected and improved version "
    "of the code."
)


class CodeAssistantNodes:
    def __init__(self, llm):
        self.llm = llm

    def planner_node(self, state: CodeAgentState) -> CodeAgentState:
        user_request = state["messages"][-1].content if state["messages"] else ""
        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=user_request),
        ]
        response = self.llm.invoke(messages)
        return {
            "messages": [response],
            "plan": response.content,
            "review_notes": "",
        }

    def coder_node(self, state: CodeAgentState) -> CodeAgentState:
        plan = state.get("plan", "")
        messages = [
            SystemMessage(content=CODER_SYSTEM),
            HumanMessage(content=f"Implementation plan:\n{plan}"),
        ]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def reviewer_node(self, state: CodeAgentState) -> CodeAgentState:
        last_code = ""
        for msg in reversed(state["messages"]):
            if hasattr(msg, "content") and msg.content:
                last_code = msg.content
                break
        messages = [
            SystemMessage(content=REVIEWER_SYSTEM),
            HumanMessage(content=f"Code to review:\n{last_code}"),
        ]
        response = self.llm.invoke(messages)
        return {
            "messages": [response],
            "review_notes": response.content,
        }
