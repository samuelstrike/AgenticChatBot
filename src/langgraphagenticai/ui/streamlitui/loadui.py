import uuid
import streamlit as st

from src.langgraphagenticai.ui.uiconfigfile import Config

TOOL_USING_USECASES = {"AI Research Assistant", "Multi-Agent Research Team"}


class LoadStreamlitUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_streamlit_ui(self):
        st.set_page_config(page_title=self.config.get_page_title(), layout="wide")
        st.header(self.config.get_page_title())

        # Persist memory thread across reruns; reset on "New Conversation"
        if "thread_id" not in st.session_state:
            st.session_state["thread_id"] = str(uuid.uuid4())
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        with st.sidebar:
            llm_options = self.config.get_llm_options()
            usecase_options = self.config.get_usecase_options()

            self.user_controls["selected_llm"] = st.selectbox("Select LLM", llm_options)

            if self.user_controls["selected_llm"] == "Groq":
                model_options = self.config.get_groq_model_options()
                self.user_controls["selected_groq_model"] = st.selectbox("Select Model", model_options)
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = st.text_input(
                    "Enter Groq API Key", type="password"
                )
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("Please enter your Groq API Key to proceed.")

            self.user_controls["selected_usecase"] = st.selectbox("Select Use Case", usecase_options)

            # Tavily key — only shown for tool-using use cases
            if self.user_controls["selected_usecase"] in TOOL_USING_USECASES:
                self.user_controls["TAVILY_API_KEY"] = st.text_input(
                    "Tavily API Key (optional — enables web search)",
                    type="password",
                    help="Leave blank to use Wikipedia only.",
                )
            else:
                self.user_controls["TAVILY_API_KEY"] = ""

            st.divider()
            if st.button("New Conversation", use_container_width=True):
                st.session_state["thread_id"] = str(uuid.uuid4())
                st.session_state["chat_history"] = []
                st.rerun()

            self.user_controls["thread_id"] = st.session_state["thread_id"]

        return self.user_controls
