import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message, thread_id="default"):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}

    def display_result_on_ui(self):
        # Render previous chat history
        for role, content in st.session_state.get("chat_history", []):
            with st.chat_message(role):
                st.markdown(content)

        with st.chat_message("user"):
            st.markdown(self.user_message)

        if self.usecase == "Basic Chatbot":
            self._display_basic_chatbot()
        elif self.usecase == "AI Research Assistant":
            self._display_research_assistant()
        elif self.usecase == "Code Assistant":
            self._display_code_assistant()
        elif self.usecase == "Multi-Agent Research Team":
            self._display_multiagent()

    # ------------------------------------------------------------------
    # Basic Chatbot
    # ------------------------------------------------------------------
    def _display_basic_chatbot(self):
        response_text = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for event in self.graph.stream(
                {"messages": [HumanMessage(content=self.user_message)]},
                config=self.config,
            ):
                for value in event.values():
                    msg = value.get("messages")
                    if msg:
                        if isinstance(msg, list):
                            msg = msg[-1]
                        if hasattr(msg, "content"):
                            response_text = msg.content
                            placeholder.markdown(response_text)

        self._save_to_history("user", self.user_message)
        self._save_to_history("assistant", response_text)

    # ------------------------------------------------------------------
    # AI Research Assistant
    # ------------------------------------------------------------------
    def _display_research_assistant(self):
        response_text = ""
        tool_calls_log = []

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.info("Researching...")

            for event in self.graph.stream(
                {"messages": [HumanMessage(content=self.user_message)]},
                config=self.config,
            ):
                for node_name, value in event.items():
                    msgs = value.get("messages", [])
                    if not isinstance(msgs, list):
                        msgs = [msgs]

                    for msg in msgs:
                        if isinstance(msg, ToolMessage):
                            tool_calls_log.append((msg.name if hasattr(msg, "name") else "tool", msg.content))
                        elif isinstance(msg, AIMessage) and msg.content:
                            response_text = msg.content

            status_placeholder.empty()

            if tool_calls_log:
                with st.expander(f"Tool calls used ({len(tool_calls_log)})", expanded=False):
                    for tool_name, result in tool_calls_log:
                        st.markdown(f"**{tool_name}**")
                        st.text(result[:500] + ("..." if len(result) > 500 else ""))

            if response_text:
                st.markdown(response_text)
            else:
                st.warning("No response generated.")

        self._save_to_history("user", self.user_message)
        self._save_to_history("assistant", response_text)

    # ------------------------------------------------------------------
    # Code Assistant
    # ------------------------------------------------------------------
    def _display_code_assistant(self):
        stage_outputs = {"planner": "", "coder": "", "reviewer": ""}

        with st.chat_message("assistant"):
            plan_expander = st.expander("Step 1: Planning", expanded=True)
            code_expander = st.expander("Step 2: Writing Code", expanded=False)
            review_expander = st.expander("Step 3: Review", expanded=False)

            for event in self.graph.stream(
                {"messages": [HumanMessage(content=self.user_message)]},
                config=self.config,
            ):
                for node_name, value in event.items():
                    msgs = value.get("messages", [])
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        if hasattr(msg, "content") and msg.content:
                            stage_outputs[node_name] = msg.content
                            if node_name == "planner":
                                plan_expander.markdown(msg.content)
                                code_expander.info("Generating code...")
                            elif node_name == "coder":
                                code_expander.markdown(msg.content)
                                review_expander.info("Reviewing code...")
                            elif node_name == "reviewer":
                                review_expander.markdown(msg.content)

        final = stage_outputs.get("reviewer") or stage_outputs.get("coder") or ""
        self._save_to_history("user", self.user_message)
        self._save_to_history("assistant", final)

    # ------------------------------------------------------------------
    # Multi-Agent Research Team
    # ------------------------------------------------------------------
    def _display_multiagent(self):
        final_response = ""
        agent_notes = []

        with st.chat_message("assistant"):
            activity_placeholder = st.empty()
            activity_placeholder.info("Supervisor is routing your request...")

            for event in self.graph.stream(
                {"messages": [HumanMessage(content=self.user_message)]},
                config=self.config,
            ):
                for node_name, value in event.items():
                    notes = value.get("agent_notes", [])
                    if notes:
                        agent_notes.extend(notes)
                        with activity_placeholder.container():
                            with st.expander("Agent Activity Log", expanded=True):
                                for note in agent_notes:
                                    st.markdown(f"- {note}")

                    msgs = value.get("messages", [])
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.content and node_name == "writer":
                            final_response = msg.content

            if final_response:
                st.markdown("---")
                st.markdown(final_response)
            elif not agent_notes:
                st.warning("No response generated.")

        self._save_to_history("user", self.user_message)
        self._save_to_history("assistant", final_response)

    # ------------------------------------------------------------------
    def _save_to_history(self, role: str, content: str):
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        st.session_state["chat_history"].append((role, content))
