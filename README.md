# Agentic AI Chatbot with LangGraph

This project is an end-to-end implementation of an agentic AI chatbot using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and [Streamlit](https://streamlit.io/). It provides a modular framework for building stateful, agentic conversational AI applications with support for multiple LLMs (currently Groq) and use cases (currently a basic chatbot).

## Features

- **Streamlit UI**: User-friendly web interface for interacting with the chatbot and configuring options.
- **LLM Integration**: Easily switch between supported LLMs and models (Groq with llama3 and gemma2 models).
- **Stateful Graph Architecture**: Uses LangGraph to manage conversational state and flow.
- **Extensible Nodes**: Modular node system for adding new use cases and capabilities.

## Project Structure

```
app.py
requirements.txt
src/
    langgraphagenticai/
        main.py
        graph/
        LLMS/
        nodes/
        state/
        ui/
```

## Getting Started

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Set Up Environment

- Obtain a [Groq API Key](https://console.groq.com/keys) if you want to use Groq LLMs.

### 3. Run the Application

```sh
streamlit run app.py
```

### 4. Using the UI

- Select your preferred LLM and model in the sidebar.
- Enter your Groq API key if required.
- Choose a use case (e.g., Basic Chatbot).
- Start chatting in the main window.

## Configuration

UI options are managed in [`src/langgraphagenticai/ui/uiconfigfile.ini`](src/langgraphagenticai/ui/uiconfigfile.ini):

```ini
[DEFAULT]
PAGE_TITLE = LangGraph: Build Stateful Agentic AI LangGraph
LLM_OPTIONS = Groq
USECASE_OPTIONS = Basic Chatbot
GROQ_MODEL_OPTIONS = llama3-8b-8192, llama3-70b-8192, gemma2-9b-it
```

## Extending the Project

- **Add new LLMs**: Implement a new class in [`src/langgraphagenticai/LLMS/`](src/langgraphagenticai/LLMS/).
- **Add new use cases**: Create new nodes in [`src/langgraphagenticai/nodes/`](src/langgraphagenticai/nodes/) and update the graph builder.

## License

This project is for educational and research purposes.

---