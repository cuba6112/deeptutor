# DeepTutor - Complete Reverse Engineering Documentation

## Overview

DeepTutor is a **Multi-Agent Teaching & Research Platform** built on a Python FastAPI backend with a Next.js 14 frontend. The system implements sophisticated AI agent pipelines for solving problems, generating questions, conducting research, and co-authoring content using RAG (Retrieval-Augmented Generation) with LightRAG/RAGAnything.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Next.js 14)                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │Dashboard│ │ Solver  │ │Research │ │Question │ │Knowledge│ ...   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
└───────┼──────────┼───────────┼───────────┼───────────┼─────────────┘
        │ WebSocket/REST API    │           │           │
┌───────▼──────────▼───────────▼───────────▼───────────▼─────────────┐
│                      BACKEND (FastAPI)                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    API Routers (11 total)                    │  │
│  │ solve │ research │ question │ knowledge │ dashboard │ ...   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                     AGENT MODULES                            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │   │
│  │  │ MainSolver  │ │  Research   │ │  Question   │ ...        │   │
│  │  │(Dual-Loop)  │ │  Pipeline   │ │  Generator  │            │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘            │   │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│  ┌──────────────────────────▼──────────────────────────────────┐   │
│  │                    CORE SERVICES                             │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────────────────┐   │   │
│  │  │ RAG Tools  │ │ KB Manager │ │ Configuration (core.py)│   │   │
│  │  │(LightRAG)  │ │            │ │                        │   │   │
│  │  └────────────┘ └────────────┘ └────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
DeepTutor/
├── src/                          # Backend Python source
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # Application entry point
│   │   ├── routers/             # API route handlers (11 routers)
│   │   └── utils/               # API utilities (history, logging, task management)
│   ├── agents/                   # AI Agent implementations
│   │   ├── solve/               # Problem solving agents (MainSolver)
│   │   ├── research/            # Research pipeline agents
│   │   ├── question/            # Question generation agents
│   │   ├── guide/               # Tutorial/guide agents
│   │   ├── ideagen/             # Idea generation agents
│   │   └── co_writer/           # Co-authoring agents
│   ├── core/                    # Core utilities
│   │   ├── core.py              # Configuration management
│   │   └── logging.py           # Unified logging system
│   ├── knowledge/               # Knowledge base management
│   │   ├── manager.py           # KnowledgeBaseManager class
│   │   ├── initializer.py       # KB initialization
│   │   ├── add_documents.py     # Document addition
│   │   └── progress_tracker.py  # Progress tracking
│   └── tools/                   # Shared tools
│       ├── rag_tool.py          # RAG query interface
│       ├── web_search.py        # Web search tool
│       ├── paper_search_tool.py # Academic paper search
│       ├── code_executor.py     # Code execution sandbox
│       └── query_item_tool.py   # Numbered item queries
├── web/                         # Frontend Next.js application
│   ├── app/                     # Next.js 14 App Router pages
│   │   ├── layout.tsx           # Root layout with GlobalProvider
│   │   ├── page.tsx             # Dashboard homepage
│   │   ├── solver/              # Problem solver page
│   │   ├── research/            # Research interface
│   │   ├── question/            # Question generator
│   │   ├── knowledge/           # Knowledge base manager
│   │   ├── notebook/            # Notebook system
│   │   ├── co_writer/           # Co-authoring editor
│   │   ├── guide/               # Tutorial guide
│   │   ├── ideagen/             # Idea generation
│   │   └── settings/            # Application settings
│   ├── components/              # Reusable React components
│   ├── context/                 # React Context providers
│   └── lib/                     # Utility functions
├── config/                      # Configuration files
│   ├── main.yaml                # Main configuration
│   ├── agents.yaml              # Agent parameters
│   └── *.yaml                   # Module-specific configs
├── data/                        # Data directory
│   ├── knowledge_bases/         # Knowledge base storage
│   └── user/                    # User output data
└── scripts/                     # Utility scripts
```

---

## Core Module (`src/core/core.py`)

The configuration management hub that loads settings from environment variables and YAML files.

### Key Functions

```python
def get_llm_config() -> dict:
    """Returns: binding, model, api_key, base_url"""
    # Environment variables: LLM_MODEL, LLM_BINDING_API_KEY, LLM_BINDING_HOST

def get_embedding_config() -> dict:
    """Returns: binding, model, api_key, base_url, dim, max_tokens"""
    # Environment variables: EMBEDDING_MODEL, EMBEDDING_BINDING_API_KEY,
    #                        EMBEDDING_BINDING_HOST, EMBEDDING_DIM

def get_tts_config() -> dict:
    """Returns: binding, model, api_key, base_url"""
    # Environment variables: TTS_MODEL, TTS_BINDING_API_KEY, TTS_BINDING_HOST

def get_agent_params(agent_name: str) -> dict:
    """Load agent parameters from config/agents.yaml"""

def load_config_with_main(config_filename: str, project_root: Path) -> dict:
    """Load config file merged with main.yaml settings"""
```

### Configuration Priority
1. `DeepTutor.env` (project-specific)
2. `.env` (fallback)
3. System environment variables

---

## API Layer (`src/api/`)

### Entry Point (`main.py`)

```python
app = FastAPI(title="DeepTutor API", version="1.0.0", lifespan=lifespan)

# Routers mounted under /api/v1
app.include_router(solve.router, prefix="/api/v1", tags=["solve"])
app.include_router(question.router, prefix="/api/v1/question", tags=["question"])
app.include_router(research.router, prefix="/api/v1/research", tags=["research"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["dashboard"])
app.include_router(co_writer.router, prefix="/api/v1/co_writer", tags=["co_writer"])
app.include_router(notebook.router, prefix="/api/v1/notebook", tags=["notebook"])
app.include_router(guide.router, prefix="/api/v1/guide", tags=["guide"])
app.include_router(ideagen.router, prefix="/api/v1/ideagen", tags=["ideagen"])
app.include_router(settings.router, prefix="/api/v1/settings", tags=["settings"])
app.include_router(system.router, prefix="/api/v1/system", tags=["system"])

# Static file serving for user outputs
app.mount("/api/outputs", StaticFiles(directory="data/user"))
```

### Key API Routers

#### 1. Solve Router (`routers/solve.py`)
- **Endpoint**: `WebSocket /api/v1/solve`
- **Purpose**: Real-time problem solving with streaming logs
- **Flow**:
  1. Accept WebSocket connection
  2. Receive JSON: `{question, kb_name}`
  3. Initialize `MainSolver` with knowledge base
  4. Stream agent status, logs, progress via WebSocket
  5. Return final answer with citations

#### 2. Knowledge Router (`routers/knowledge.py`)
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /list` - List all knowledge bases
  - `GET /{kb_name}` - Get KB details
  - `DELETE /{kb_name}` - Delete KB
  - `POST /create` - Create new KB with files
  - `POST /{kb_name}/upload` - Upload files to existing KB
  - `GET /{kb_name}/progress` - Get initialization progress
  - `WebSocket /{kb_name}/progress/ws` - Real-time progress updates

#### 3. Research Router (`routers/research.py`)
- **Endpoints**:
  - `POST /optimize_topic` - Rephrase/optimize research topic
  - `WebSocket /run` - Run full research pipeline
- **Parameters**: `topic`, `kb_name`, `plan_mode` (quick/medium/deep/auto), `enabled_tools`

#### 4. System Router (`routers/system.py`)
- **Endpoints**:
  - `GET /status` - System status (backend, LLM, embeddings, TTS)
  - `POST /test/llm` - Test LLM connection
  - `POST /test/embeddings` - Test embedding model
  - `POST /test/tts` - Test TTS configuration

---

## Agent System

### MainSolver (`src/agents/solve/main_solver.py`)

The **Dual-Loop Architecture** for problem solving:

```
┌─────────────────────────────────────────────────────────────────┐
│                     ANALYSIS LOOP                                │
│  ┌──────────────┐     ┌──────────────┐                          │
│  │ Investigate  │ ──► │    Note      │  (Extract key info)      │
│  │    Agent     │     │    Agent     │                          │
│  └──────────────┘     └──────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SOLVE LOOP                                  │
│  ┌─────────┐   ┌────────┐   ┌───────┐   ┌───────┐   ┌────────┐ │
│  │ Manager │──►│ Solve  │──►│ Tool  │──►│ Check │──►│Response│ │
│  │  Agent  │   │ Agent  │   │ Agent │   │ Agent │   │ Agent  │ │
│  └─────────┘   └────────┘   └───────┘   └───────┘   └────────┘ │
│                                                            │     │
│                              ┌────────────────────┐        │     │
│                              │ PrecisionAnswer    │◄───────┘     │
│                              │      Agent         │              │
│                              └────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

#### Agent Responsibilities

| Agent | Role |
|-------|------|
| `InvestigateAgent` | Initial question analysis, identify key concepts |
| `NoteAgent` | Extract and organize investigation findings |
| `ManagerAgent` | Orchestrate solving strategy |
| `SolveAgent` | Execute solution steps |
| `ToolAgent` | Execute RAG queries, web search, code |
| `ResponseAgent` | Format intermediate responses |
| `CheckAgent` | Verify solution quality |
| `PrecisionAnswerAgent` | Generate final precise answer |

#### Memory Systems
- `InvestigateMemory`: Stores investigation context
- `SolveMemory`: Tracks solving progress
- `CitationMemory`: Manages source citations

### Research Pipeline (`src/agents/research/research_pipeline.py`)

Three-stage workflow: **Planning → Researching → Reporting**

```python
class ResearchPipeline:
    """Coordinates research workflow using DynamicTopicQueue"""

    agents = {
        "rephrase": RephraseAgent,    # Optimize research topic
        "decompose": DecomposeAgent,  # Break into subtopics
        "manager": ManagerAgent,      # Coordinate research
        "research": ResearchAgent,    # Execute research queries
        "note": NoteAgent,            # Take research notes
        "reporting": ReportingAgent,  # Generate final report
    }
```

#### Plan Modes
| Mode | Subtopics | Iterations | Description |
|------|-----------|------------|-------------|
| `quick` | 2 | 2 | Fast overview |
| `medium` | 5 | 4 | Balanced depth |
| `deep` | 8 | 7 | Comprehensive |
| `auto` | auto | flexible | AI-determined |

---

## Knowledge Base System

### KnowledgeBaseManager (`src/knowledge/manager.py`)

```python
class KnowledgeBaseManager:
    """Manages multiple knowledge bases with centralized configuration"""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.config_file = base_dir / "kb_config.json"

    def list_knowledge_bases(self) -> list[str]
    def get_info(self, kb_name: str) -> dict  # name, statistics, is_default
    def delete_knowledge_base(self, kb_name: str, confirm: bool) -> bool
    def get_rag_storage_path(self, kb_name: str) -> Path
```

### Knowledge Base Structure

```
data/knowledge_bases/{kb_name}/
├── raw/                 # Original documents (PDF, TXT, etc.)
├── images/              # Extracted images
├── rag_storage/         # LightRAG vector store
│   ├── vdb_entities/    # Entity embeddings
│   ├── vdb_chunks/      # Chunk embeddings
│   └── *.json           # Graph data
├── content_list/        # Extracted numbered items
└── metadata.json        # KB metadata
```

### Initialization (`src/knowledge/initializer.py`)

```python
class KnowledgeBaseInitializer:
    """Initialize new knowledge base with document processing"""

    async def process_documents(self):
        """Process all documents using RAGAnything"""
        # 1. Parse documents (PDF, TXT, DOCX, etc.)
        # 2. Extract text and images
        # 3. Build knowledge graph via LightRAG
        # 4. Generate embeddings

    def extract_numbered_items(self):
        """Extract numbered lists, equations, definitions"""
```

---

## RAG Integration (`src/tools/rag_tool.py`)

### rag_search Function

```python
async def rag_search(
    query: str,
    kb_name: str = "ai_textbook",
    mode: str = "hybrid",         # hybrid, local, global, naive
    only_need_context: bool = True,
    top_k: int = 10,
    max_token_for_text_unit: int = 2000,
    max_token_for_global_context: int = 2000,
    max_token_for_local_context: int = 2000,
) -> str:
    """
    Query knowledge base using RAGAnything/LightRAG

    Modes:
    - hybrid: Combines local + global context
    - local: Entity-focused retrieval
    - global: Community-level summaries
    - naive: Direct chunk retrieval
    """
```

### Configuration Flow

```python
# Get embedding config from environment
embedding_config = get_embedding_config()
embedding_dim = embedding_config["dim"]      # e.g., 768 for Ollama
embedding_model = embedding_config["model"]  # e.g., embeddinggemma:300m

# Create EmbeddingFunc wrapper
embedding_func = EmbeddingFunc(
    embedding_dim=embedding_dim,
    max_token_size=embedding_max_tokens,
    func=lambda texts: openai_embed(texts, model=embedding_model, ...)
)

# Initialize RAGAnything
rag = RAGAnything(
    config=RAGAnythingConfig(working_dir=rag_storage_dir),
    llm_model_func=llm_model_func,
    embedding_func=embedding_func
)
```

---

## Frontend Architecture

### Technology Stack
- **Framework**: Next.js 14 (App Router)
- **Styling**: Tailwind CSS with dark mode
- **State**: React Context (`GlobalProvider`)
- **Icons**: Lucide React

### Root Layout (`web/app/layout.tsx`)

```tsx
export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <GlobalProvider>
          <div className="flex h-screen">
            <Sidebar />
            <main className="flex-1 overflow-y-auto">
              {children}
            </main>
          </div>
        </GlobalProvider>
      </body>
    </html>
  );
}
```

### Pages

| Page | Route | Purpose |
|------|-------|---------|
| Dashboard | `/` | Activity feed, stats, quick actions |
| Solver | `/solver` | Problem solving interface |
| Research | `/research` | Deep research with reports |
| Question | `/question` | Quiz/question generation |
| Knowledge | `/knowledge` | KB management |
| Notebook | `/notebook` | Saved work organization |
| Co-Writer | `/co_writer` | Collaborative writing |
| Guide | `/guide` | Tutorial generation |
| IdeaGen | `/ideagen` | Idea brainstorming |
| Settings | `/settings` | App configuration |

### Key Components

```
web/components/
├── Sidebar.tsx              # Navigation sidebar
├── SystemStatus.tsx         # Backend/LLM status display
├── ActivityDetail.tsx       # Activity detail modal
├── CoWriterEditor.tsx       # Rich text editor
├── Mermaid.tsx             # Diagram rendering
├── research/               # Research-specific components
│   ├── ResearchDashboard.tsx
│   ├── TaskGrid.tsx
│   └── ActiveTaskDetail.tsx
├── question/               # Question-specific components
│   ├── QuestionDashboard.tsx
│   └── QuestionTaskGrid.tsx
└── ui/                     # Base UI components
    ├── Button.tsx
    └── Modal.tsx
```

---

## Data Flow

### Problem Solving Flow

```
User Input (Question)
       │
       ▼
┌──────────────────┐
│ Frontend (React) │  WebSocket connect to /api/v1/solve
└────────┬─────────┘
         │ {question, kb_name}
         ▼
┌──────────────────┐
│  Solve Router    │  Initialize MainSolver
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  MainSolver      │
│  ┌────────────┐  │
│  │ Analysis   │  │  InvestigateAgent → NoteAgent
│  │   Loop     │  │
│  └────────────┘  │
│        │         │
│        ▼         │
│  ┌────────────┐  │
│  │  Solve     │  │  Manager → Solve → Tool → Check → Response
│  │   Loop     │  │
│  └────────────┘  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   RAG Tools      │  Query knowledge base
│ (rag_tool.py)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  RAGAnything/    │  Vector search + knowledge graph
│   LightRAG       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ PrecisionAnswer  │  Generate final response
│     Agent        │
└────────┬─────────┘
         │
         ▼ WebSocket: {type: "result", final_answer, metadata}
┌──────────────────┐
│ Frontend Display │
└──────────────────┘
```

### Knowledge Base Creation Flow

```
User Uploads Files
       │
       ▼
┌──────────────────┐
│ POST /create     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Create Directory │  raw/, images/, rag_storage/, content_list/
│    Structure     │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ Background Task      │  (run_initialization_task)
│                      │
│  ┌────────────────┐  │
│  │ Process Docs   │  │  RAGAnything parses PDFs, extracts text
│  └────────┬───────┘  │
│           ▼          │
│  ┌────────────────┐  │
│  │ Build KG       │  │  LightRAG builds knowledge graph
│  └────────┬───────┘  │
│           ▼          │
│  ┌────────────────┐  │
│  │ Generate       │  │  Create embeddings for entities/chunks
│  │ Embeddings     │  │
│  └────────┬───────┘  │
│           ▼          │
│  ┌────────────────┐  │
│  │ Extract Items  │  │  Numbered lists, equations, definitions
│  └────────────────┘  │
└──────────────────────┘
         │
         ▼ WebSocket: Progress updates
┌──────────────────┐
│ Frontend Status  │
└──────────────────┘
```

---

## Configuration Files

### Environment Variables (`.env` / `DeepTutor.env`)

```bash
# LLM Configuration
LLM_MODEL=minimax-m2.1:cloud
LLM_BINDING_API_KEY=ollama
LLM_BINDING_HOST=http://localhost:11434/v1

# Embedding Configuration
EMBEDDING_MODEL=embeddinggemma:300m
EMBEDDING_BINDING_API_KEY=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434/v1
EMBEDDING_DIM=768

# TTS Configuration (optional)
TTS_MODEL=tts-1
TTS_BINDING_API_KEY=your_key
TTS_BINDING_HOST=https://api.openai.com/v1
```

### Main Configuration (`config/main.yaml`)

```yaml
paths:
  user_data_dir: "./data/user"
  user_log_dir: "./data/user/logs"
  knowledge_bases_dir: "./data/knowledge_bases"

logging:
  log_dir: "./data/user/logs"
  level: INFO

research:
  planning:
    decompose:
      initial_subtopics: 5
      max_subtopics: 10
  researching:
    max_iterations: 5
    enable_rag_hybrid: true
    enable_web_search: true
```

### Agent Parameters (`config/agents.yaml`)

```yaml
solve:
  investigate:
    max_iterations: 3
    temperature: 0.7
  manager:
    temperature: 0.5
  solve:
    max_steps: 10

research:
  rephrase:
    temperature: 0.8
  decompose:
    max_depth: 3
```

---

## External Dependencies

### Python Dependencies (key ones)
- `fastapi` / `uvicorn` - Web framework
- `lightrag` - Knowledge graph RAG
- `raganything` - Multi-modal document processing
- `openai` - LLM API client (OpenAI-compatible)
- `pydantic` - Data validation
- `websockets` - Real-time communication

### Frontend Dependencies
- `next` (14.x) - React framework
- `tailwindcss` - Styling
- `lucide-react` - Icons
- `react-markdown` - Markdown rendering
- `mermaid` - Diagram rendering

---

## Important Implementation Details

### 1. Embedding Dimension Handling

LightRAG's `openai_embed` decorator hardcodes `embedding_dim=1536`. When using custom models (e.g., Ollama with 768-dim embeddings), wrap with `EmbeddingFunc`:

```python
embedding_func = EmbeddingFunc(
    embedding_dim=768,  # Actual dimension from config
    func=lambda texts: openai_embed.func(
        texts, model=model, api_key=api_key, base_url=base_url
    )
)
```

### 2. WebSocket Streaming Pattern

All long-running operations use WebSocket for real-time updates:

```python
@router.websocket("/solve")
async def websocket_solve(websocket: WebSocket):
    await websocket.accept()

    # Log queue for streaming
    log_queue = asyncio.Queue()

    # Background pusher
    async def log_pusher():
        while not done:
            entry = await log_queue.get()
            await websocket.send_json(entry)

    # Run operation with log interception
    result = await solver.solve(question)
    await websocket.send_json({"type": "result", ...})
```

### 3. Task ID Management

Centralized task tracking with `TaskIDManager`:

```python
task_manager = TaskIDManager.get_instance()
task_id = task_manager.generate_task_id("solve", task_key)
task_manager.update_task_status(task_id, "completed")
```

### 4. Progress Tracking

File-based progress for long operations:

```python
progress_tracker = ProgressTracker(kb_name, base_dir)
progress_tracker.update(
    ProgressStage.PROCESSING_DOCUMENTS,
    "Processing files...",
    current=5,
    total=10
)
```

---

## Security Considerations

1. **API Keys**: Stored in environment variables, never committed
2. **File Access**: Limited to `data/` directory
3. **CORS**: Currently allows all origins (development mode)
4. **Input Validation**: Pydantic models for request validation

---

## Performance Optimizations

1. **Lazy Initialization**: KB manager initialized on first request
2. **Background Tasks**: Long operations run in FastAPI `BackgroundTasks`
3. **Caching**: LightRAG caches embeddings and LLM responses
4. **Async**: Full async/await pattern throughout

---

## Debugging Tips

1. **Check logs**: `data/user/logs/ai_tutor_YYYYMMDD.log`
2. **API health**: `GET /api/v1/knowledge/health`
3. **System status**: `GET /api/v1/system/status`
4. **Test connections**: `POST /api/v1/system/test/llm`, `/test/embeddings`
5. **Progress tracking**: Check `kb_progress.json` files in KB directories

---

## Extending the System

### Adding a New Agent Module

1. Create directory: `src/agents/{new_agent}/`
2. Implement agent classes with `process()` method
3. Create API router: `src/api/routers/{new_agent}.py`
4. Register router in `src/api/main.py`
5. Create frontend page: `web/app/{new_agent}/page.tsx`
6. Add navigation in `Sidebar.tsx`

### Adding a New Tool

1. Create tool in `src/tools/{tool_name}.py`
2. Implement async function with docstring for agent usage
3. Register in agent's tool list

---

*Document generated through reverse engineering analysis of the DeepTutor codebase.*
