import streamlit as st
from langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from langgraphagenticai.LLMS.groqllm import GroqLLM
from langgraphagenticai.graph.graph_builder import GraphBuilder
from langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit
from langgraph.checkpoint.memory import MemorySaver


def load_langgraph_agenticai_app():
    """
    Load and runs the LangGraph Agentic AI application using Streamlit UI.
    """

    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    user_message = st.chat_input("Enter your message:")

    if user_message:
        try:
            obj_llm_config = GroqLLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_model()

            if not model:
                st.error("Error: Failed to initialize the LLM model.")
                return

            usecase = user_input.get("selected_usecase")
            if not usecase:
                st.error("Error: No use case selected.")
                return

            thread_id = user_input.get("thread_id", "default")
            tavily_api_key = user_input.get("TAVILY_API_KEY", "")

            # Persist the MemorySaver in session_state so memory survives reruns
            if "checkpointer" not in st.session_state:
                st.session_state["checkpointer"] = MemorySaver()

            graph_builder = GraphBuilder(
                model=model,
                checkpointer=st.session_state["checkpointer"],
            )

            try:
                graph = graph_builder.setup_graph(usecase, tavily_api_key=tavily_api_key)
                DisplayResultStreamlit(
                    usecase, graph, user_message, thread_id
                ).display_result_on_ui()

            except Exception as e:
                st.error(f"Error: Failed to build the graph for use case '{usecase}': {e}")
                return

        except Exception as e:
            st.error(f"Error: Graph initialization failed: {e}")
            return
