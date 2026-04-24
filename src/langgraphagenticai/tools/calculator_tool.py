from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a safe mathematical expression. Input must be a valid Python math expression (e.g. '2 + 2 * 10')."""
    allowed = set("0123456789+-*/()., ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions are allowed."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"
