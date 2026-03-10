import streamlit as st
import uuid
import os
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from src.graph import app as agent_graph

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GitLab Security Agent",
    page_icon="🦊",
    layout="centered"
)

# Custom Styling for a professional corporate look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🦊 GitLab Security Policy Agent")
st.info("Agentic RAG System: Self-Correction")

# --- INITIALIZE SESSION STATE ---
# Track chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = []

# Persistent thread_id for LangGraph's MemorySaver (Persistence)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0

if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0

if "last_inference_time" not in st.session_state:
    st.session_state.last_inference_time = None

if "last_node_path" not in st.session_state:
    st.session_state.last_node_path = []

# --- SIDEBAR ---
with st.sidebar:
    st.header("System Status")
    st.success("Vector Store: Connected")
    st.success("LLM: Groq (Llama-3.1-8b)")
    
    st.divider()
    st.subheader("📊 Session Metrics")
    st.metric("Total Queries", st.session_state.total_queries)
    avg_time = (
        st.session_state.total_time / st.session_state.total_queries
        if st.session_state.total_queries > 0 else 0.0
    )
    st.metric("Avg Inference Time", f"{avg_time:.2f}s")
    if st.session_state.last_inference_time is not None:
        st.metric("Last Query Time", f"{st.session_state.last_inference_time:.2f}s")
    if st.session_state.last_node_path:
        st.divider()
        st.caption("Last Agent Path")
        st.code(" → ".join(st.session_state.last_node_path), language=None)
    st.divider()
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.total_queries = 0
        st.session_state.total_time = 0.0
        st.session_state.last_inference_time = None
        st.session_state.last_node_path = []
        st.rerun()

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- CHAT INPUT & EXECUTION ---
if prompt := st.chat_input("Ask a policy question..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Agent Execution
    with st.chat_message("assistant"):
        # Use st.status to show the "Self-Correction" steps to the user in real-time
        with st.status("Agent searching policies...", expanded=True) as status:
            
            # Setup the graph config with our persistent thread_id
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Initial State for the Graph
            initial_state = {"messages": [HumanMessage(content=prompt)]}
            
            final_response = ""
            node_path = []

            # --- Inference Timer Start ---
            start_time = time.time()

            # Single stream pass using "updates" mode — gives node name + state changes
            for event in agent_graph.stream(initial_state, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name not in node_path:
                        node_path.append(node_name)
                        st.write(f"`{node_name}` complete")
                    # Capture final AI response
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if isinstance(msg, AIMessage):
                                final_response = msg.content

            # --- Inference Timer End ---
            elapsed = time.time() - start_time

            # Update session metrics
            st.session_state.last_inference_time = elapsed
            st.session_state.total_time += elapsed
            st.session_state.total_queries += 1
            st.session_state.last_node_path = node_path

            status.update(
                label=f"Done in {elapsed:.2f}s | Path: {' → '.join(node_path)}",
                state="complete",
                expanded=False
            )

        # Display the final answer with Citations
        st.markdown(final_response)
        
        # Save to session state
        st.session_state.messages.append({"role": "assistant", "content": final_response})

# --- FOOTER ---
st.markdown("---")
st.caption("Built with LangGraph, Groq, and ChromaDB • Final Project Milestone")