from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


def get_search_tools(tavily_api_key: str = "") -> list:
    """Return available search tools. Tavily is included only when an API key is provided."""
    tools = []

    if tavily_api_key:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            import os
            os.environ["TAVILY_API_KEY"] = tavily_api_key
            tools.append(TavilySearchResults(max_results=3))
        except Exception:
            pass

    wikipedia = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
    )
    tools.append(wikipedia)

    return tools
