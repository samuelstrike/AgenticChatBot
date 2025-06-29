import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self, user_controls_input):
        self.user_controls = user_controls_input

    def get_llm_model(self):
        try:
            groq_api_key = self.user_controls.get("GROQ_API_KEY")
            selected_groq_model = self.user_controls.get("selected_groq_model")

            if groq_api_key =='' and os.environ['GROQ_API_KEY'] == '':
                st.error("Please enter your Groq API Key to proceed.")
            
            llm = ChatGroq(api_key=groq_api_key, model=selected_groq_model)

        except Exception as e:
            raise ValueError(f"Failed to initialize Groq LLM: {e}")
        
        return llm