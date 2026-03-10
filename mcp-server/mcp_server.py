import sys
import os
import logging
from pathlib import Path

# Suppress noisy model loading progress bars in MCP stderr output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP
from langchain_core.messages import HumanMessage, AIMessage

from src.graph import app as agent_graph
from src.config import Config

mcp = FastMCP("Agentic Doc Search")


# 1. Resource: Expose the list of available policy documents
@mcp.resource("policies://list")
def list_policies() -> str:
    """Returns the names of all policy documents loaded into the RAG system."""
    policy_dir = Config.DATA_PATH
    if not os.path.exists(policy_dir):
        return "No policy documents found."
    files = [f for f in os.listdir(policy_dir) if f.endswith(".md")]
    return "\n".join(sorted(files))


# 2. Tool: The Agentic Search Engine
@mcp.tool()
def search_security_policies(query: str) -> str:
    """
    Search through GitLab security and technology policies using a
    self-corrective agentic RAG system. Returns a cited, accurate answer
    grounded in the policy documents. If the query is out of scope,
    returns a polite rejection instead of hallucinating.

    Args:
        query: The question to ask about the security policies.
    """
    # AgentState expects messages as a list of LangChain message objects
    initial_state = {"messages": [HumanMessage(content=query)]}

    # Use a fixed thread_id for MCP sessions (stateless per call)
    config = {"configurable": {"thread_id": "mcp_session"}}

    result = agent_graph.invoke(initial_state, config=config)

    # Extract the last AIMessage from the messages list
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content

    return "No response generated."


if __name__ == "__main__":
    mcp.run()
