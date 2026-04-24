import operator
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class CodeAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    plan: str
    review_notes: str


class MultiAgentState(TypedDict):
    messages: Annotated[List, add_messages]
    next_agent: str
    agent_notes: Annotated[List[str], operator.add]
