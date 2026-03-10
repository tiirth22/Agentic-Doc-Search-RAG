from typing_extensions import TypedDict , Literal , Annotated , Sequence
from typing import Optional
from langchain_core.messages import BaseMessage 
from langgraph.graph.message import add_messages


class AgentState(TypedDict) :
    """
    The state of our Agentic RAG system.
    Supports the ReAct pattern: Thought -> Action -> Observation -> Final Answer.
    """
    
    # It ensures new messages are appended to the conversation history.
    messages : Annotated[Sequence[BaseMessage] , add_messages]

    # Store the actual documents retrieved for 'Self-Correction' or 'Grading'
    documents : list[str]

    # Store metadata (policy_title, filename) for each retrieved chunk
    doc_metadata : Optional[list[dict]]

    # Constraint to only allow specific values (Prevents logic bugs)
    next_action : Literal["retrieve", "generate", "end", "no_answer"] 

