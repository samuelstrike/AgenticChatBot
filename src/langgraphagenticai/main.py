import streamlit as st
from langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI

def load_langgraph_agenticai_app():
    """
    Load and runs the LangGraph Agentic AI application using Streamlit UI.
    """

    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Fail to load user input from the UI.")

        return
    
    user_message = st.chat_input("Enter your message:")

   