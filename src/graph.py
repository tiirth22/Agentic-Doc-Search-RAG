from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.config import Config
from src.vector_store import VectorStoreManager
from src.state import AgentState
from src.prompts import RAG_PROMPT

# Initialize Components
llm = ChatGroq(
    api_key=Config.GROQ_API_KEY, 
    model=Config.LLM_MODEL_NAME, 
    temperature=Config.TEMPERATURE
)

vs_manager = VectorStoreManager()
retriever = vs_manager.get_retriever()

# 2. Nodes
def retrieve_node(state: AgentState):
    print("--- NODE: RETRIEVING DOCUMENTS ---")
    last_message = state['messages'][-1].content
    docs = retriever.invoke(last_message)

    return {
        "documents": [doc.page_content for doc in docs],
        "doc_metadata": [doc.metadata for doc in docs]
    }

def grade_documents_node(state: AgentState):
    """
    Determines whether the retrieved documents are relevant to the user's question.
    """
    print("--- NODE: CHECKING DOCUMENT RELEVANCE ---")
    last_user_message = state["messages"][-1].content
    docs = state["documents"]
    
    # Use plain text grading to avoid structured output failures on smaller models
    grader_llm = ChatGroq(
        api_key=Config.GROQ_API_KEY,
        model=Config.LLM_MODEL_NAME,
        temperature=Config.TEMPERATURE
    )

    system_prompt = (
        "You are a grader assessing relevance of a retrieved document to a user question. "
        "If the document contains keywords or content related to the user question, grade it as relevant. "
        "It does not need to be a stringent test. The goal is to filter out completely unrelated retrievals. "
        "Respond with ONLY a single word: 'yes' if relevant, or 'no' if not relevant. No explanation."
    )

    doc_text = "\n".join(docs)
    response = grader_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User question: {last_user_message}\n\nRetrieved document:\n{doc_text}")
    ])

    binary_score = response.content.strip().lower().split()[0] if response.content.strip() else "no"

    if binary_score == "yes":
        print("--- DECISION: DOCUMENT RELEVANT ---")
        return {"documents": docs, "next_action": "generate"}
    else:
        print("--- DECISION: DOCUMENT NOT RELEVANT ---")
        return {"documents": [], "next_action": "no_answer"}

def no_answer_node(state: AgentState):
    print("--- NODE: NO RELEVANT DOCUMENTS FOUND ---")
    from langchain_core.messages import AIMessage
    response = AIMessage(content=(
        "I'm sorry, I don't have enough information in my knowledge base to answer that question. "
        "My knowledge is limited to GitLab's Security and Technology Policies. "
        "Please try asking something related to access management, audit logging, "
        "change management, penetration testing, software development lifecycle, or policy management."
    ))
    return {"messages": [response], "next_action": "end"}

def generate_node(state: AgentState):
    print("--- NODE: GENERATING RESPONSE ---")
    docs = state["documents"]
    metadata = state.get("doc_metadata", [{}] * len(docs))

    # Build context with policy title headers so LLM knows the source of each chunk
    context_parts = []
    for i, (text, meta) in enumerate(zip(docs, metadata)):
        title = meta.get("policy_title", "Unknown Policy")
        filename = meta.get("filename", "")
        context_parts.append(f"[Source: {title} | File: {filename}]\n{text}")
    context = "\n\n---\n\n".join(context_parts)
    
    # We pass the context into our professional RAG prompt
    system_message = SystemMessage(content=RAG_PROMPT.format(context=context))
    
    # ReAct: Logic + History
    messages = [system_message] + list(state["messages"])
    response = llm.invoke(messages)
    
    return {
        "messages": [response],
        "next_action": "end"
    }

# --- BUILD GRAPH ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade_documents", grade_documents_node)
workflow.add_node("generate", generate_node)
workflow.add_node("no_answer", no_answer_node)

# --- Define Edges ---
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Self-Correction Loop
workflow.add_conditional_edges(
    "grade_documents",
    lambda x: x["next_action"],
    {
        "generate": "generate",
        "retrieve": "retrieve",
        "no_answer": "no_answer"
    }
)

workflow.add_edge("no_answer", END)

workflow.add_edge("generate", END)

# Compile with Checkpointer for Chat History
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)