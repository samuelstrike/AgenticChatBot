# Agentic AI Chatbot (Multiagent System)

An end-to-end implementation of a multiagent AI chatbot using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Streamlit](https://streamlit.io/). The project demonstrates stateful agentic AI with tool calling, multi-step reasoning, persistent memory, and a supervisor-based multiagent architecture.

## Features

- **Persistent Memory**: Conversations are remembered within a session using LangGraph's `MemorySaver` checkpointer. Start fresh anytime with the "New Conversation" button.
- **Tool Calling**: Agents autonomously search the web (Tavily) or Wikipedia to answer research questions.
- **Multi-Step Reasoning**: The Code Assistant pipeline makes each reasoning stage (plan → code → review) visible in the UI.
- **Multiagent System**: A Supervisor agent dynamically routes tasks between a Researcher, Analyst, and Writer — each a specialised sub-agent.
- **Streamlit UI**: Sidebar for LLM/model/API key/use-case selection with live agent activity logs and expandable intermediate steps.
- **Extensible Architecture**: Modular nodes, tools, and graph builders make adding new use cases straightforward.

## Use Cases

| Use Case | Description | Key Concepts |
|---|---|---|
| **Basic Chatbot** | Stateful conversational AI with memory | LangGraph state, MemorySaver |
| **AI Research Assistant** | Autonomously searches the web to answer questions | ReAct loop, tool calling, Tavily/Wikipedia |
| **Code Assistant** | Plans, writes, and reviews code in three explicit stages | Multi-step reasoning pipeline, structured state |
| **Multi-Agent Research Team** | Supervisor routes work between Researcher, Analyst, and Writer agents | Supervisor pattern, multiagent routing, structured output |

## Project Structure

```
app.py
requirements.txt
src/
    langgraphagenticai/
        main.py
        graph/
            graph_builder.py       # All 4 graph topologies
        LLMS/
            groqllm.py
        nodes/
            basic_chatbot_node.py
            research_assistant_nodes.py
            code_assistant_nodes.py
            multiagent_nodes.py
        state/
            state.py               # Base message state
            agent_state.py         # CodeAgentState, MultiAgentState
        tools/
            search_tools.py        # Tavily + Wikipedia
            calculator_tool.py
        ui/
            uiconfigfile.ini
            streamlitui/
                loadui.py
                display_result.py
```

## Getting Started

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. API Keys

| Key | Required | Where to get |
|---|---|---|
| Groq API Key | Yes | [console.groq.com/keys](https://console.groq.com/keys) |
| Tavily API Key | Optional | [app.tavily.com](https://app.tavily.com) — enables web search; Wikipedia is used as fallback |

### 3. Run the Application

```sh
streamlit run app.py
```

### 4. Using the UI

1. Select your LLM and model in the sidebar (use `llama-3.3-70b-versatile` for best results with tool calling and structured output).
2. Enter your Groq API Key.
3. Choose a use case.
4. Optionally add a Tavily API Key for web search (Research Assistant and Multi-Agent Team).
5. Start chatting — intermediate reasoning steps appear as expandable sections.

## Configuration

UI options are managed in `src/langgraphagenticai/ui/uiconfigfile.ini`:

```ini
[DEFAULT]
PAGE_TITLE = Agentic AI Chatbot (Multiagent System)
LLM_OPTIONS = Groq
USECASE_OPTIONS = Basic Chatbot, AI Research Assistant, Code Assistant, Multi-Agent Research Team
GROQ_MODEL_OPTIONS = llama-3.3-70b-versatile, llama-3.1-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it
```

## Extending the Project

- **Add a new LLM**: Implement a new class in `src/langgraphagenticai/LLMS/` and add it to `LLM_OPTIONS`.
- **Add a new use case**: Create a node file in `nodes/`, add a graph method in `graph_builder.py`, and register it in `uiconfigfile.ini`.
- **Persistent storage**: Swap `MemorySaver` for `SqliteSaver` in `main.py` for cross-session persistence.

## License

This project is for educational and research purposes.

---
