# 🦊 Agentic Doc Search RAG

> **An intelligent document search system powered by Agentic RAG — it doesn't just retrieve, it *thinks* before answering.**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-Agent-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Groq-Llama_3.1-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_Store-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-UI-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/MCP-Server-blueviolet?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

---

## 📸 Screenshots

| Accurate Policy Answer | Out-of-Scope Rejection |
|:---:|:---:|
| ![screenshot1](assets/SS1.png) | ![screenshot2](assets/SS2.png) |

| MCP Inspector Validation |
|:---:|
| ![screenshot4](assets/SS4.png) |

---

## 💡 What is this?

Most RAG systems follow a simple pattern: take a user's question, fetch some documents, and generate an answer. The problem? They *always* answer — even when the retrieved documents have nothing to do with the question. This leads to hallucinations that look convincing but are completely wrong.

I built this project to tackle that exact issue. Instead of a basic retrieve-and-generate pipeline, this system uses an **agentic approach** — it evaluates whether the retrieved documents are actually relevant before deciding to answer. If they're not relevant, it says so honestly instead of making something up.

The knowledge base I'm using here is a set of **6 GitLab Security & Technology Policy documents** (Access Management, Audit Logging, Change Management, Penetration Testing, SDLC, and Policy Governance).

---

## 🧠 How it Works

The whole system is built as a **LangGraph state graph** with 4 nodes that work together:

```
  User asks a question
          │
          ▼
    ┌───────────┐
    │  Retrieve  │   → Searches ChromaDB for the top-6 most relevant chunks
    └─────┬─────┘
          │
          ▼
  ┌───────────────┐
  │  Grade Docs   │   → LLM checks: "Are these documents actually relevant?"
  └───────┬───────┘
          │
     ┌────┴────┐
     │         │
  Relevant   Not Relevant
     │         │
     ▼         ▼
 ┌────────┐  ┌──────────┐
 │Generate│  │ No Answer │  → Politely declines instead of hallucinating
 └────┬───┘  └─────┬────┘
      │            │
      └─────┬──────┘
            ▼
          Done
```

### Auto-generated Graph from LangGraph

![Agent Graph](agent_graph.png)

> Generated using `graph.get_graph().draw_mermaid_png()` — this is the real compiled graph, not a hand-drawn diagram.

---

## ⚙️ Key Technical Choices

Here are some of the decisions I made while building this and why:

| What | Choice | Reasoning |
|---|---|---|
| **Framework** | LangGraph StateGraph | Needed conditional branching — a regular LangChain chain can't skip nodes or route dynamically |
| **Relevance Check** | Separate LLM grading step | The grading happens *before* generation, so irrelevant docs never make it to the answer step |
| **Grading Format** | Plain yes/no text | Llama 3.1 8B doesn't handle structured JSON output reliably, simple text works consistently |
| **When docs aren't relevant** | Dedicated `no_answer` node | Instead of looping or retrying, it gives an honest "I don't know" — prevents hallucination |
| **How citations work** | Each chunk gets a `[Source: Policy \| File:]` header | The LLM sees the source info right in its context, so it always knows where facts come from |
| **Search strategy** | Cosine similarity (not MMR) | MMR was pulling in chunks from unrelated policies for diversity — pure similarity is more accurate here |
| **Conversation memory** | LangGraph MemorySaver | Keeps chat history per session using `thread_id` — supports multi-turn conversations |
| **Embeddings** | `all-MiniLM-L6-v2` (local) | Runs offline, no API costs, no rate limits — good enough for this corpus size |
| **LLM** | Groq Llama 3.1 8B Instant | Sub-second responses, free tier works for development |
| **MCP Integration** | FastMCP server | Makes the RAG pipeline callable by external AI agents like Claude Desktop or VS Code Copilot |

---

## 🔧 Implementation Details

### Data Ingestion

Policy documents (`.md` files) go through this pipeline:

1. **Load** — `UnstructuredMarkdownLoader` reads each file while preserving its structure
2. **Clean** — Strip out markdown formatting artifacts and extra whitespace  
3. **Split** — `RecursiveCharacterTextSplitter` with `chunk_size=800` and `overlap=150` so information at chunk boundaries isn't lost
4. **Tag** — Each chunk gets metadata like `{ "policy_title": "Audit Logging Policy", "filename": "audit-logging-policy.md" }`
5. **Store** — Chunks are embedded using `all-MiniLM-L6-v2` and saved to ChromaDB

### Agent State

All nodes share a typed state:

```python
class AgentState(TypedDict):
    messages:      Annotated[list[AnyMessage], add_messages]
    documents:     Optional[list[Document]]
    doc_metadata:  Optional[list[dict]]
    next_action:   Optional[Literal["generate", "no_answer"]]
```

The `doc_metadata` field tracks which policy each chunk came from — this is what makes accurate citations possible.

### The Nodes

- **`retrieve_node`** — Embeds the query, fetches 6 closest chunks from ChromaDB, stores documents + metadata in state
- **`grade_documents_node`** — Asks the LLM "are these docs relevant to the question?" and sets `next_action` to either `generate` or `no_answer`
- **`generate_node`** — Builds a formatted context with source headers for each chunk, then generates a cited answer
- **`no_answer_node`** — Returns a polite message saying it couldn't find relevant info

### Routing Logic

```python
graph.add_conditional_edges(
    "grade_documents",
    lambda state: state["next_action"],
    {"generate": "generate", "no_answer": "no_answer"}
)
```

### Streamlit UI

The app streams graph execution in real-time using `stream_mode="updates"`, so users can see each step as it happens:
- Live agent path display (`retrieve → grade_documents → generate`)
- Inference time tracking
- Session metrics in the sidebar (query count, average response time, last path taken)

---

## 🌐 MCP Server

The RAG agent is also exposed as an **MCP (Model Context Protocol) server**, so external AI tools can query the policy documents directly.

**What's exposed:**

| Type | Name | What it does |
|---|---|---|
| Tool | `search_security_policies` | Accepts a question, runs the full agent graph, returns the answer |
| Resource | `policies://list` | Returns names of all indexed policy documents |

**Run it:**
```bash
cd mcp-server
python mcp_server.py
```

**Test with MCP Inspector:**
```bash
npx @modelcontextprotocol/inspector@0.14.3 python mcp_server.py
```

---

## 📁 Project Structure

```
Agentic-Doc-Search-RAG/
├── app.py                    # Streamlit chat interface
├── save_graph.py             # Generates the agent graph diagram
├── agent_graph.png           # Visual representation of the agent flow
├── pyproject.toml            # Dependencies
├── .env                      # API keys (not committed)
│
├── assets/                   # Screenshots
│
├── data/
│   └── security-and-technology-policies/
│       ├── access-management-policy.md
│       ├── audit-logging-policy.md
│       ├── change-management-policy.md
│       ├── penetration-testing-policy.md
│       ├── software-development-lifecycle-policy.md
│       └── security-and-technology-policies-management.md
│
├── mcp-server/
│   └── mcp_server.py        # FastMCP server
│
└── src/
    ├── config.py             # Environment variables and settings
    ├── state.py              # AgentState definition
    ├── graph.py              # Node implementations + graph compilation
    ├── prompts.py            # System prompts for the LLM
    ├── data_ingestion.py     # Document loading and chunking
    ├── vector_store.py       # ChromaDB setup and retriever
    ├── schema.py             # Data schemas
    └── utils.py              # Helper functions
```

---

## 🚀 Getting Started

### 1. Clone and enter the project
```bash
git clone https://github.com/tiirth22/Agentic-Doc-Search-RAG.git
cd Agentic-Doc-Search-RAG
```

### 2. Set up a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e .
```

### 4. Add your API key
Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
You can get a free key from [console.groq.com](https://console.groq.com).

### 5. Initialize the vector store (first time only)
```bash
python -c "
from src.data_ingestion import DataIngestor
from src.vector_store import VectorStoreManager
ingestor = DataIngestor()
chunks = ingestor.load_and_split()
VectorStoreManager().create_vector_store(chunks)
"
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## 📊 Example Queries

| Query | What happens | Result |
|---|---|---|
| "Who is responsible for implementing the Audit Logging Policy?" | `retrieve → grade → generate` | Correctly identifies the Security Team |
| "What system tiers are in scope for Change Management?" | `retrieve → grade → generate` | Lists Tiers 1-3 as in-scope, notes Tier 4 is excluded |
| "How often must penetration tests be conducted?" | `retrieve → grade → generate` | At minimum annually + after significant system changes |
| "How do I reset my GitLab password?" | `retrieve → grade → no_answer` | Recognizes this isn't in the documents, declines gracefully |
| "What is the company's vacation policy?" | `retrieve → grade → no_answer` | Completely out of scope — handled without any hallucination |

---

## 🔍 What's Different from a Standard RAG?

| Traditional RAG | This Project |
|---|---|
| Always produces an answer | Only answers when documents are relevant |
| Can silently hallucinate | Routes to an honest "I don't know" |
| No visibility into the process | Shows the full agent path in real time |
| No source tracking | Every chunk is tagged with its source document |
| One-shot queries only | Multi-turn chat with conversation memory |
| Only works through one UI | Also available as an MCP tool for AI agents |

---

## 🐛 Bugs I Ran Into (and Fixed)

A few issues I hit during development that might help if you're building something similar:

1. **Structured output failures** — Llama 3.1 8B kept failing when I asked for JSON responses during grading. Switched to plain `yes/no` text and it worked reliably.

2. **Environment variables not loading** — Had to use `override=True` in `load_dotenv()` because without it, the library skips variables that are already set in the system environment.

3. **State key mismatch** — My grading node was returning `{"generate": "yes"}` but the router expected `state["next_action"]`. Took a while to debug that `KeyError`.

4. **MMR returning wrong documents** — The diversity-first approach of MMR was pulling chunks from unrelated policies. Switching to standard similarity search fixed the accuracy.

5. **Metadata access in generation** — Instead of digging into LangChain `Document.metadata` during generation, I store `doc_metadata` as its own state field during retrieval. Cleaner and less error-prone.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for the full text.

---

<p align="center">
  <b>Built by Tirth Jignesh Dalal</b><br/>
  <a href="https://github.com/tiirth22">GitHub</a>
</p>
