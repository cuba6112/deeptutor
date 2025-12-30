# DeepTutor RAG System Documentation

## Overview

DeepTutor implements a sophisticated **Retrieval-Augmented Generation (RAG)** system built on top of **LightRAG** and **RAGAnything**. The RAG system enables intelligent querying of knowledge bases with multiple retrieval modes, knowledge graph construction, and multimodal document processing.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      rag_search() [rag_tool.py]                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │ LLM Config      │  │ Embedding Config│  │ KB Path         │          │
│  │ (get_llm_config)│  │ (get_embedding) │  │ Resolution      │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
└───────────┼────────────────────┼────────────────────┼────────────────────┘
            │                    │                    │
            ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RAGAnything Instance                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    LightRAG Core Engine                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │    │
│  │  │ Vector Store│  │ Knowledge   │  │ Text Chunks             │  │    │
│  │  │ (Embeddings)│  │ Graph       │  │ Storage                 │  │    │
│  │  │             │  │ (Entities + │  │                         │  │    │
│  │  │             │  │  Relations) │  │                         │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       QUERY MODES                                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │
│  │  hybrid   │  │   local   │  │  global   │  │   naive   │            │
│  │ (default) │  │           │  │           │  │           │            │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RETRIEVED CONTEXT + LLM ANSWER                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. RAG Tool (`src/tools/rag_tool.py`)

The main interface for querying knowledge bases.

#### Function Signature

```python
async def rag_search(
    query: str,
    kb_name: str | None = None,
    mode: str = "hybrid",
    api_key: str | None = None,
    base_url: str | None = None,
    kb_base_dir: str | None = None,
    **kwargs,
) -> dict:
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | The search query |
| `kb_name` | str | None | Knowledge base name (uses default if not specified) |
| `mode` | str | "hybrid" | Query mode: "hybrid", "local", "global", "naive" |
| `api_key` | str | None | LLM API key (from environment if not provided) |
| `base_url` | str | None | LLM API base URL (from environment if not provided) |
| `kb_base_dir` | str | None | Knowledge base directory path |
| `only_need_context` | bool | False | Return only retrieved context without LLM generation |
| `only_need_prompt` | bool | False | Return only the constructed prompt |

#### Return Value

```python
{
    "query": str,      # Original query
    "answer": str,     # Retrieved/generated answer
    "mode": str        # Query mode used
}
```

---

### 2. Query Modes

DeepTutor supports **4 distinct query modes**, each optimized for different use cases:

| Mode | Description | Use Case | Strategy |
|------|-------------|----------|----------|
| **hybrid** | Balanced retrieval combining vector search and knowledge graph | Default, comprehensive answers | Vector similarity + Entity/Relation graph traversal |
| **local** | Entity-focused retrieval | Specific concept queries | Retrieves local entity neighborhoods from knowledge graph |
| **global** | Global context retrieval | Broad topic questions | Uses global knowledge graph relationships and community summaries |
| **naive** | Simple vector search | Fast retrieval, basic queries | Pure semantic similarity search on text chunks |

#### Mode Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                   QUERY MODE DECISION TREE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Need comprehensive answer?                                      │
│      └── YES → Use "hybrid" (default)                           │
│      └── NO  ↓                                                  │
│                                                                  │
│  Looking for specific entity/concept?                           │
│      └── YES → Use "local"                                      │
│      └── NO  ↓                                                  │
│                                                                  │
│  Need broad topic overview?                                     │
│      └── YES → Use "global"                                     │
│      └── NO  ↓                                                  │
│                                                                  │
│  Need fast, simple retrieval?                                   │
│      └── YES → Use "naive"                                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 3. Knowledge Base Manager (`src/knowledge/manager.py`)

Manages multiple knowledge bases with centralized configuration.

#### Key Methods

```python
class KnowledgeBaseManager:
    def __init__(self, base_dir="./data/knowledge_bases")

    def list_knowledge_bases(self) -> list[str]
    def get_knowledge_base_path(self, name: str | None) -> Path
    def get_rag_storage_path(self, name: str | None) -> Path
    def get_info(self, name: str | None) -> dict
    def set_default(self, name: str)
    def get_default(self) -> str | None
    def delete_knowledge_base(self, name: str, confirm: bool) -> bool
```

#### Knowledge Base Structure

```
data/knowledge_bases/
├── kb_config.json                    # Master configuration (tracks all KBs)
└── {kb_name}/
    ├── metadata.json                 # KB metadata (name, created_at, version)
    ├── raw/                          # Original documents (PDF, TXT, DOCX, MD)
    ├── images/                       # Extracted images from documents
    ├── content_list/                 # Document content hierarchies (JSON)
    ├── numbered_items.json           # Extracted definitions, theorems, formulas
    └── rag_storage/                  # LightRAG internal storage
        ├── kv_store_full_entities.json    # Entity data
        ├── kv_store_full_relations.json   # Relation data
        ├── kv_store_text_chunks.json      # Text chunk data
        └── (vector indices and graph files)
```

---

### 4. Knowledge Base Initializer (`src/knowledge/initializer.py`)

Processes documents and builds the RAG knowledge graph.

#### Initialization Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  KNOWLEDGE BASE INITIALIZATION                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Create Directory Structure                             │
│      └── raw/, images/, rag_storage/, content_list/             │
│      └── metadata.json, kb_config.json registration             │
│                                                                  │
│  Step 2: Copy Documents                                         │
│      └── Copy source files to raw/ directory                    │
│      └── Supported: .pdf, .docx, .doc, .txt, .md                │
│                                                                  │
│  Step 3: Process Documents (RAGAnything)                        │
│      └── Parse documents (MinerU for PDFs)                      │
│      └── Extract text, images, tables, equations                │
│      └── Build content hierarchy (chapters → sections)          │
│                                                                  │
│  Step 4: Build Knowledge Graph (LightRAG)                       │
│      └── Extract entities (concepts, definitions)               │
│      └── Identify relationships between entities                │
│      └── Store text chunks with embeddings                      │
│                                                                  │
│  Step 5: Extract Numbered Items                                 │
│      └── Definitions, Theorems, Lemmas, Propositions           │
│      └── Corollaries, Examples, Remarks, Figures               │
│      └── Equations, Tables                                      │
│      └── Save to numbered_items.json                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Configuration

```python
config = RAGAnythingConfig(
    working_dir=str(rag_storage_dir),
    enable_image_processing=True,
    enable_table_processing=True,
    enable_equation_processing=True,
)
```

---

### 5. Embedding Configuration

DeepTutor uses configurable embeddings through environment variables.

#### Environment Variables

```bash
# Embedding Configuration
EMBEDDING_BINDING=openai              # Service provider
EMBEDDING_MODEL=text-embedding-3-large # Model name
EMBEDDING_BINDING_API_KEY=sk-...      # API key
EMBEDDING_BINDING_HOST=https://api.openai.com/v1  # Base URL
EMBEDDING_DIM=3072                    # Embedding dimension
EMBEDDING_MAX_TOKENS=8192             # Max tokens for embedding
```

#### EmbeddingFunc Wrapper

```python
embedding_func = EmbeddingFunc(
    embedding_dim=embedding_dim,        # e.g., 3072 for text-embedding-3-large
    max_token_size=embedding_max_tokens, # e.g., 8192
    func=lambda texts: openai_embed(
        texts,
        model=embedding_model,
        api_key=embedding_api_key,
        base_url=embedding_base_url,
    ),
)
```

> **Important**: When using custom embedding models (e.g., Ollama with 768-dim embeddings), ensure the `EMBEDDING_DIM` matches your model's actual output dimension.

---

### 6. Query Item Tool (`src/tools/query_item_tool.py`)

Enables exact lookup of numbered items (definitions, theorems, equations, etc.).

#### Function Signature

```python
def query_numbered_item(
    identifier: str,
    kb_name: str | None = None,
    kb_base_dir: str | None = None,
    max_results: int | None = None,
) -> dict:
```

#### Supported Item Types

| Type | Example Identifiers |
|------|---------------------|
| Definition | "Definition 1.1", "Definition 2.3" |
| Theorem | "Theorem 2.1", "Theorem 3.5" |
| Lemma | "Lemma 1.2", "Lemma 4.1" |
| Formula/Equation | "(1.2.1)", "(2.3.5)" |
| Figure | "Figure 1.1", "Figure 2.5" |
| Example | "Example 1.1", "Example 3.2" |
| Remark | "Remark 2.1", "Remark 4.3" |

#### Matching Priority

1. **Exact match** (highest priority)
2. **Case-insensitive exact match**
3. **Prefix match** (e.g., "2.1" matches "(2.1.1)", "(2.1.2)")
4. **Partial match** (contains query string)

---

## Configuration

### Main Configuration (`config/main.yaml`)

```yaml
tools:
  rag_tool:
    kb_base_dir: ./data/knowledge_bases
    default_kb: ai_textbook
  query_item:
    enabled: true
    max_results: 5
  web_search:
    enabled: true

research:
  rag:
    kb_name: DE-all
    default_mode: hybrid
    fallback_mode: naive
  researching:
    enable_rag_naive: true
    enable_rag_hybrid: true

question:
  rag_mode: naive
  rag_query_count: 3
```

### LLM Configuration

```bash
# LLM Configuration (.env)
LLM_MODEL=gpt-4o
LLM_BINDING_API_KEY=sk-...
LLM_BINDING_HOST=https://api.openai.com/v1
```

---

## Data Flow

### Query Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  rag_search()   │ ◄── Load LLM Config
│                 │ ◄── Load Embedding Config
│                 │ ◄── Resolve KB Path
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RAGAnything    │
│  Instance       │
│  ┌───────────┐  │
│  │ LightRAG  │  │
│  └───────────┘  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              Query Mode Selection                │
│                                                  │
│  hybrid → Vector Search + Graph Traversal       │
│  local  → Entity Neighborhood Search            │
│  global → Global Graph Context                  │
│  naive  → Pure Vector Similarity Search         │
│                                                  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│            Retrieved Context                     │
│                                                  │
│  if only_need_context=True:                     │
│      → Return context directly                  │
│  else:                                          │
│      → LLM generates answer with context        │
│                                                  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Return: {query, answer, mode}                  │
└─────────────────────────────────────────────────┘
```

### Knowledge Base Creation Flow

```
Upload Documents
    │
    ▼
┌─────────────────┐
│ POST /create    │  (API endpoint)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Create Directory│  raw/, images/, rag_storage/, content_list/
│ Structure       │
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────┐
│           Background Processing Task            │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │ 1. Parse Documents (RAGAnything)        │   │
│  │    • PDF → MinerU parser                │   │
│  │    • Extract text, images, tables       │   │
│  └─────────────────────────────────────────┘   │
│                      │                          │
│                      ▼                          │
│  ┌─────────────────────────────────────────┐   │
│  │ 2. Build Knowledge Graph (LightRAG)     │   │
│  │    • Extract entities                   │   │
│  │    • Identify relationships             │   │
│  │    • Store in kv_store_*.json          │   │
│  └─────────────────────────────────────────┘   │
│                      │                          │
│                      ▼                          │
│  ┌─────────────────────────────────────────┐   │
│  │ 3. Generate Embeddings                  │   │
│  │    • Embed entities                     │   │
│  │    • Embed text chunks                  │   │
│  │    • Build vector indices               │   │
│  └─────────────────────────────────────────┘   │
│                      │                          │
│                      ▼                          │
│  ┌─────────────────────────────────────────┐   │
│  │ 4. Extract Numbered Items               │   │
│  │    • LLM-assisted identification        │   │
│  │    • Save to numbered_items.json        │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ KB Ready for    │
│ Queries         │
└─────────────────┘
```

---

## Usage Examples

### Basic RAG Query

```python
from src.tools.rag_tool import rag_search

# Simple query with default settings
result = await rag_search(
    query="What is attention mechanism in transformers?",
    kb_name="ai_textbook",
    mode="hybrid"
)

print(result["answer"])
```

### Context-Only Retrieval

```python
# Get only the retrieved context without LLM generation
result = await rag_search(
    query="Explain backpropagation",
    kb_name="deep_learning",
    mode="naive",
    only_need_context=True
)

context = result["answer"]  # Raw retrieved context
```

### Query Numbered Items

```python
from src.tools.query_item_tool import query_numbered_item

# Query a specific definition
result = query_numbered_item(
    identifier="Definition 2.3",
    kb_name="ai_textbook"
)

if result["status"] == "success":
    print(f"Found: {result['content']}")
```

### Initialize New Knowledge Base

```python
from src.knowledge.initializer import KnowledgeBaseInitializer

initializer = KnowledgeBaseInitializer(
    kb_name="my_textbook",
    base_dir="./data/knowledge_bases",
    api_key="your-api-key",
    base_url="https://api.openai.com/v1"
)

# Create structure and process documents
initializer.create_directory_structure()
initializer.copy_documents(["chapter1.pdf", "chapter2.pdf"])
await initializer.process_documents()
initializer.extract_numbered_items()
```

### CLI Usage

```bash
# Initialize a new knowledge base
python src/knowledge/initializer.py my_kb --docs document.pdf

# List all knowledge bases
python src/knowledge/manager.py list

# Get knowledge base info
python src/knowledge/manager.py info my_kb

# Set default knowledge base
python src/knowledge/manager.py set-default my_kb
```

---

## Integration with Agents

### Research Pipeline Integration

```python
from src.tools.rag_tool import rag_search

# During research decomposition phase
async def gather_background_knowledge(topic, kb_name):
    result = await rag_search(
        query=f"Background information about: {topic}",
        kb_name=kb_name,
        mode="hybrid"
    )
    return result["answer"]
```

### Question Generation Integration

```python
# Retrieve context for question generation
result = await rag_search(
    query=query,
    kb_name=self.kb_name,
    mode="naive",
    only_need_context=True
)
context = result["answer"]
```

### Solve Agent Integration

```python
# Query knowledge base during problem solving
answer = await rag_search(
    query=question,
    kb_name=kb_name,
    mode="hybrid"
)
```

---

## RAG Storage Files

### Entity Storage (`kv_store_full_entities.json`)

```json
{
    "entity_id_1": {
        "name": "Attention Mechanism",
        "type": "concept",
        "description": "...",
        "embedding": [0.1, 0.2, ...]
    }
}
```

### Relation Storage (`kv_store_full_relations.json`)

```json
{
    "relation_id_1": {
        "source": "entity_id_1",
        "target": "entity_id_2",
        "type": "is_part_of",
        "description": "..."
    }
}
```

### Text Chunks Storage (`kv_store_text_chunks.json`)

```json
{
    "chunk_id_1": {
        "content": "The attention mechanism allows...",
        "source": "document.pdf",
        "page": 15,
        "embedding": [0.1, 0.2, ...]
    }
}
```

---

## Advanced Features

### 1. Multimodal Processing

RAGAnything supports processing of:
- **Images**: Extracted and stored in `images/` directory
- **Tables**: Parsed and included in knowledge graph
- **Equations**: Mathematical formulas extracted and indexed

### 2. Incremental Document Addition

Add documents to existing knowledge bases without reprocessing:

```python
from src.knowledge.add_documents import DocumentAdder

adder = DocumentAdder(kb_name="my_kb")
await adder.add_documents(["new_document.pdf"])
```

### 3. Progress Tracking

Real-time progress tracking via `ProgressTracker`:

```python
from src.knowledge.progress_tracker import ProgressTracker, ProgressStage

tracker = ProgressTracker(kb_name, base_dir)
tracker.update(
    ProgressStage.PROCESSING_DOCUMENTS,
    "Processing file 3 of 10...",
    current=3,
    total=10
)
```

### 4. Fallback Modes

Configure fallback modes when primary retrieval fails:

```yaml
research:
  rag:
    default_mode: hybrid
    fallback_mode: naive  # Fallback if hybrid fails
```

---

## Troubleshooting

### Common Issues

1. **"RAG storage not found" error**
   - Ensure knowledge base is initialized
   - Run: `python src/knowledge/initializer.py <kb_name> --docs <files>`

2. **Embedding dimension mismatch**
   - Check `EMBEDDING_DIM` matches your model's output
   - Default for `text-embedding-3-large`: 3072

3. **Knowledge base not in config**
   - Register manually: Edit `kb_config.json`
   - Or reinitialize the knowledge base

4. **Query returns empty results**
   - Verify documents were processed successfully
   - Check `rag_storage/` contains populated JSON files
   - Try different query modes (naive, local, global)

### Debugging

```bash
# Check system status
GET /api/v1/system/status

# Test LLM connection
POST /api/v1/system/test/llm

# Test embeddings
POST /api/v1/system/test/embeddings

# Check KB health
GET /api/v1/knowledge/health

# View logs
tail -f data/user/logs/ai_tutor_*.log
```

---

## Performance Considerations

1. **Lazy Initialization**: KB manager initialized on first request
2. **Caching**: LightRAG caches embeddings and LLM responses
3. **Async Operations**: Full async/await pattern for concurrent queries
4. **Batch Processing**: Documents processed in batches during initialization
5. **Parallel Execution**: Research supports parallel RAG queries (up to 5)

---

## Related Files

| File | Purpose |
|------|---------|
| [src/tools/rag_tool.py](src/tools/rag_tool.py) | Main RAG query interface |
| [src/tools/query_item_tool.py](src/tools/query_item_tool.py) | Numbered item lookup |
| [src/knowledge/manager.py](src/knowledge/manager.py) | KB management |
| [src/knowledge/initializer.py](src/knowledge/initializer.py) | KB initialization |
| [src/knowledge/add_documents.py](src/knowledge/add_documents.py) | Incremental document addition |
| [config/main.yaml](config/main.yaml) | Main configuration |

---

*Document generated for DeepTutor RAG System v0.1*
