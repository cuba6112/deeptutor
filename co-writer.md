# Co-Writer Module - Reverse Engineering Documentation

This document provides a complete reverse-engineered explanation of the DeepTutor Co-Writer module, including how to duplicate it using Ollama.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Details](#component-details)
3. [Data Flow](#data-flow)
4. [Prompts Reference](#prompts-reference)
5. [API Endpoints](#api-endpoints)
6. [Ollama Integration Guide](#ollama-integration-guide)
7. [Minimal Standalone Implementation](#minimal-standalone-implementation)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Co-Writer Module                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Router                           │   │
│  │                 (src/api/routers/co_writer.py)               │   │
│  │                                                              │   │
│  │  POST /edit      POST /automark    POST /narrate            │   │
│  │  POST /export    GET /history      GET /tts/status          │   │
│  └──────────────┬────────────────────────┬─────────────────────┘   │
│                 │                        │                          │
│     ┌───────────▼───────────┐   ┌───────▼───────────┐              │
│     │     EditAgent         │   │   NarratorAgent   │              │
│     │ (edit_agent.py)       │   │(narrator_agent.py)│              │
│     │                       │   │                   │              │
│     │ - process()           │   │ - generate_script()              │
│     │ - auto_mark()         │   │ - generate_audio()│              │
│     │                       │   │ - narrate()       │              │
│     └───────────┬───────────┘   └───────┬───────────┘              │
│                 │                        │                          │
│     ┌───────────▼───────────────────────▼───────────┐              │
│     │            LightRAG OpenAI Client              │              │
│     │        (openai_complete_if_cache)              │              │
│     └───────────────────────┬───────────────────────┘              │
│                             │                                       │
│     ┌───────────────────────▼───────────────────────┐              │
│     │     LLM Backend (OpenAI / Ollama / etc.)      │              │
│     │                                                │              │
│     │  Ollama: http://localhost:11434/v1             │              │
│     │  OpenAI: https://api.openai.com/v1             │              │
│     └────────────────────────────────────────────────┘              │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    External Tools                             │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │ RAG Search │  │ Web Search │  │ TTS (OpenAI/Ollama)    │  │  │
│  │  │(rag_tool)  │  │(web_search)│  │                        │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|------|---------|
| `src/agents/co_writer/edit_agent.py` | Text editing (rewrite/shorten/expand) with RAG/Web context |
| `src/agents/co_writer/narrator_agent.py` | Script generation and TTS audio creation |
| `src/api/routers/co_writer.py` | FastAPI REST endpoints |
| `src/agents/co_writer/prompts/en/*.yaml` | English prompts for agents |
| `src/agents/co_writer/prompts/zh/*.yaml` | Chinese prompts for agents |
| `config/agents.yaml` | Agent parameters (temperature, max_tokens) |

---

## Component Details

### 1. EditAgent (`edit_agent.py`)

The EditAgent handles all text editing operations with optional context enhancement from RAG or web search.

#### Core Method: `process()`

```python
async def process(
    text: str,                                    # Original text to edit
    instruction: str,                             # User's editing instruction
    action: Literal["rewrite", "shorten", "expand"] = "rewrite",
    source: Literal["rag", "web"] | None = None,  # Optional context source
    kb_name: str | None = None,                   # Knowledge base name for RAG
) -> dict[str, Any]
```

#### Flow:

1. **Context Retrieval** (optional):
   - If `source="rag"`: Calls `rag_search()` with the instruction as query
   - If `source="web"`: Calls `web_search()` with the instruction

2. **Prompt Construction**:
   ```
   System: "You are an expert editor and writing assistant."

   User: "{action_verb} the following text based on the user's instruction.
         User Instruction: {instruction}

         [Reference Context: {context}]  # Only if source was provided

         Target Text to Edit:
         {text}

         Output only the edited text, without quotes or explanations."
   ```

3. **LLM Call**: Uses `openai_complete_if_cache()` from LightRAG

4. **History Recording**: Saves operation to `data/user/co-writer/history.json`

#### Core Method: `auto_mark()`

Adds semantic annotation tags to text for highlighting key information.

```python
async def auto_mark(text: str) -> dict[str, Any]
```

Uses HTML-like annotations:
- `<span data-rough-notation="circle">` - Key terms (max 5 chars)
- `<span data-rough-notation="highlight">` - Definitions
- `<span data-rough-notation="box">` - Formulas/data
- `<span data-rough-notation="underline">` - Conclusions
- `<span data-rough-notation="bracket">` - Critical paragraphs

---

### 2. NarratorAgent (`narrator_agent.py`)

Converts text content into narration scripts and optionally generates TTS audio.

#### Core Method: `generate_script()`

```python
async def generate_script(
    content: str,                               # Note content (Markdown)
    style: str = "friendly"                     # friendly/academic/concise
) -> dict[str, Any]
```

**Styles:**
- `friendly`: Conversational tutor style ("we", "us", "you")
- `academic`: Formal lecture style ("this paper", "the author")
- `concise`: Direct, efficient knowledge delivery

**Output limited to 4000 characters** (OpenAI TTS limit is 4096)

#### Core Method: `generate_audio()`

```python
async def generate_audio(
    script: str,
    voice: str = None                          # alloy/echo/fable/onyx/nova/shimmer
) -> dict[str, Any]
```

Uses OpenAI TTS API (or compatible endpoint):
```python
client = OpenAI(base_url=tts_config["base_url"], api_key=tts_config["api_key"])
response = client.audio.speech.create(
    model=tts_config["model"],
    voice=voice,
    input=script
)
response.stream_to_file(audio_path)
```

#### Combined Method: `narrate()`

```python
async def narrate(
    content: str,
    style: str = "friendly",
    voice: str = None,
    skip_audio: bool = False
) -> dict[str, Any]
```

Combines script generation + audio generation in one call.

---

## Data Flow

### Edit Flow

```
User Request
     │
     ▼
┌─────────────────┐
│ POST /edit      │
│                 │
│ {text,          │
│  instruction,   │
│  action,        │
│  source,        │
│  kb_name}       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ EditAgent       │────▶│ RAG/Web Search  │ (optional)
│ .process()      │◀────│ Context         │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Build Prompt    │
│ System + User   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ LLM Call        │
│ (Ollama/OpenAI) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save History    │
│ Return Result   │
└─────────────────┘
```

### Narration Flow

```
User Request
     │
     ▼
┌─────────────────┐
│ POST /narrate   │
│                 │
│ {content,       │
│  style,         │
│  voice,         │
│  skip_audio}    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ NarratorAgent   │
│ .narrate()      │
└────────┬────────┘
         │
         ├──────────────────────┐
         ▼                      ▼
┌─────────────────┐    ┌─────────────────┐
│ generate_script │    │ extract_key_    │
│ (LLM Call)      │    │ points (LLM)    │
└────────┬────────┘    └────────┬────────┘
         │                      │
         ▼                      │
┌─────────────────┐             │
│ generate_audio  │◀────────────┘
│ (TTS API)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Save Audio File │
│ Return URLs     │
└─────────────────┘
```

---

## Prompts Reference

### EditAgent Prompts (`prompts/en/edit_agent.yaml`)

#### System Prompt
```yaml
system: |
  You are an expert editor and writing assistant.
```

#### Action Template
```yaml
action_template: |
  {action_verb} the following text based on the user's instruction.

  User Instruction: {instruction}
```

#### Context Template (when RAG/Web used)
```yaml
context_template: |
  Reference Context:
  {context}
```

#### User Template
```yaml
user_template: |
  Target Text to Edit:
  {text}

  Output only the edited text, without quotes or explanations.
```

#### Auto-Mark System Prompt
```yaml
auto_mark_system: |
  You are a professional academic reading annotation assistant...

  ## Available Tags and Precise Usage Scenarios

  ### 1. Circle - Use Sparingly
  <span data-rough-notation="circle">content</span>
  Applicable: Core topic words, proper nouns, key metrics

  ### 2. Highlight - Moderate Use
  <span data-rough-notation="highlight">content</span>
  Applicable: Definitions, core concepts

  ### 3. Box - Minimal Use
  <span data-rough-notation="box">content</span>
  Applicable: Formulas, equations, code

  ### 4. Underline - Moderate Use
  <span data-rough-notation="underline">content</span>
  Applicable: Conclusions, key arguments

  ### 5. Bracket - Use Sparingly
  <span data-rough-notation="bracket">content</span>
  Applicable: Critical paragraphs, summaries
```

### NarratorAgent Prompts (`prompts/en/narrator_agent.yaml`)

#### Style: Friendly
```yaml
style_friendly: |
  You are a friendly and approachable tutor...
  1. Use "we", "us", "you" to create closeness
  2. Relaxed but professional tone
  3. Appropriate pauses with "well", "next", "so"
  4. Highlight with "this is important", "note here"
```

#### Style: Academic
```yaml
style_academic: |
  You are a senior scholar giving an academic lecture...
  1. Use "we", "this paper" academic language
  2. Rigorous and professional tone
  3. Clear introduction-body-conclusion structure
```

#### Style: Concise
```yaml
style_concise: |
  You are an efficient knowledge communicator...
  1. Direct and to the point
  2. General first, then details
  3. Use "first", "then", "finally"
```

#### Script Generation System
```yaml
generate_script_system_template: |
  You are a professional note narration script writing expert.

  {style_prompt}
  {length_instruction}

  **Output Format**:
  Output the narration script text directly.

  **Notes**:
  1. Maintain core information and logical structure
  2. Convert Markdown to oral descriptions
  3. Describe math formulas orally
  4. **Control length within 4000 characters**
```

---

## API Endpoints

### Edit Text
```http
POST /api/v1/co_writer/edit
Content-Type: application/json

{
  "text": "Original text to edit",
  "instruction": "Make it more formal",
  "action": "rewrite",           // rewrite | shorten | expand
  "source": "rag",               // null | rag | web
  "kb_name": "my_knowledge_base" // required if source=rag
}

Response:
{
  "edited_text": "Edited text result",
  "operation_id": "20250101_120000_abc123"
}
```

### Auto-Mark Text
```http
POST /api/v1/co_writer/automark
Content-Type: application/json

{
  "text": "Text to annotate with semantic markers"
}

Response:
{
  "marked_text": "<span data-rough-notation=\"highlight\">Annotated</span> text",
  "operation_id": "20250101_120000_abc123"
}
```

### Generate Narration
```http
POST /api/v1/co_writer/narrate
Content-Type: application/json

{
  "content": "Note content to narrate",
  "style": "friendly",           // friendly | academic | concise
  "voice": "alloy",              // alloy | echo | fable | onyx | nova | shimmer
  "skip_audio": false
}

Response:
{
  "script": "Generated narration script...",
  "key_points": ["Key point 1", "Key point 2"],
  "style": "friendly",
  "original_length": 500,
  "script_length": 450,
  "has_audio": true,
  "audio_url": "/api/outputs/co-writer/audio/narration_xxx.mp3",
  "audio_id": "20250101_120000_abc123",
  "voice": "alloy"
}
```

---

## Ollama Integration Guide

### Environment Configuration

Create or update your `.env` file:

```bash
# LLM Configuration for Ollama
LLM_BINDING=openai
LLM_MODEL=llama3.2              # or any Ollama model
LLM_BINDING_API_KEY=ollama      # Ollama doesn't need a real key
LLM_BINDING_HOST=http://localhost:11434/v1

# Embedding Configuration (optional, for RAG)
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_BINDING_API_KEY=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434/v1
EMBEDDING_DIM=768

# TTS Configuration (optional, for narration audio)
# Note: Ollama doesn't have TTS - use OpenAI or skip audio
TTS_MODEL=tts-1
TTS_API_KEY=your_openai_key
TTS_URL=https://api.openai.com/v1
TTS_VOICE=alloy
```

### How It Works with Ollama

The Co-Writer uses `openai_complete_if_cache()` from LightRAG, which is compatible with any OpenAI-compatible API. Ollama provides an OpenAI-compatible endpoint at `/v1`.

**Key compatibility points:**

1. **Endpoint**: Ollama exposes `http://localhost:11434/v1/chat/completions`
2. **API Key**: Can be any non-empty string (e.g., "ollama")
3. **Model**: Must match an installed Ollama model name

### Verify Ollama is Working

```bash
# List available models
ollama list

# Test chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## Minimal Standalone Implementation

Here's a complete standalone implementation of the Co-Writer EditAgent using Ollama:

### `cowriter_ollama.py`

```python
#!/usr/bin/env python
"""
Minimal Co-Writer implementation using Ollama
Can be run independently without the full DeepTutor system
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
import uuid

from openai import AsyncOpenAI

# Configuration - adjust for your setup
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "llama3.2"  # or any model you have installed
OLLAMA_API_KEY = "ollama"  # Ollama ignores this but OpenAI client requires it

# Output directory
OUTPUT_DIR = Path("./cowriter_output")
OUTPUT_DIR.mkdir(exist_ok=True)


class OllamaCoWriter:
    """Minimal Co-Writer implementation using Ollama"""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        api_key: str = OLLAMA_API_KEY,
    ):
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.history: list[dict] = []

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Call Ollama via OpenAI-compatible API"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    async def edit(
        self,
        text: str,
        instruction: str,
        action: Literal["rewrite", "shorten", "expand"] = "rewrite",
        context: str | None = None,
    ) -> dict[str, Any]:
        """
        Edit text based on instruction

        Args:
            text: Original text to edit
            instruction: User's editing instruction
            action: Type of edit (rewrite/shorten/expand)
            context: Optional additional context

        Returns:
            Dict with edited_text and operation_id
        """
        operation_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        # System prompt
        system_prompt = "You are an expert editor and writing assistant."

        # Build user prompt
        action_verbs = {"rewrite": "Rewrite", "shorten": "Shorten", "expand": "Expand"}
        action_verb = action_verbs.get(action, "Rewrite")

        user_prompt = f"""{action_verb} the following text based on the user's instruction.

User Instruction: {instruction}

"""
        if context:
            user_prompt += f"""Reference Context:
{context}

"""
        user_prompt += f"""Target Text to Edit:
{text}

Output only the edited text, without quotes or explanations."""

        # Call LLM
        edited_text = await self._call_llm(system_prompt, user_prompt)

        # Record history
        record = {
            "id": operation_id,
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "input": {"text": text, "instruction": instruction},
            "output": {"edited_text": edited_text},
            "model": self.model,
        }
        self.history.append(record)

        # Save to file
        history_file = OUTPUT_DIR / "history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

        return {"edited_text": edited_text, "operation_id": operation_id}

    async def auto_mark(self, text: str) -> dict[str, Any]:
        """
        Add semantic annotation tags to text

        Args:
            text: Text to annotate

        Returns:
            Dict with marked_text and operation_id
        """
        operation_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        system_prompt = """You are a professional academic reading annotation assistant.

## Task
Read the input text and carefully select the most critical information for annotation.

## Available Tags

1. Circle - Key terms (max 5 chars):
   <span data-rough-notation="circle">content</span>

2. Highlight - Definitions, core concepts:
   <span data-rough-notation="highlight">content</span>

3. Box - Formulas, data, code:
   <span data-rough-notation="box">content</span>

4. Underline - Conclusions, key arguments:
   <span data-rough-notation="underline">content</span>

5. Bracket - Critical paragraphs:
   <span data-rough-notation="bracket">content</span>

## Rules
- Annotation density should not exceed 10% of total text
- Do not modify original text, only insert HTML tags
- If no annotation needed, return text as-is"""

        user_prompt = f"""Process the following text:
{text}"""

        marked_text = await self._call_llm(system_prompt, user_prompt)

        record = {
            "id": operation_id,
            "timestamp": datetime.now().isoformat(),
            "action": "automark",
            "input": {"text": text},
            "output": {"marked_text": marked_text},
            "model": self.model,
        }
        self.history.append(record)

        return {"marked_text": marked_text, "operation_id": operation_id}

    async def generate_narration_script(
        self,
        content: str,
        style: Literal["friendly", "academic", "concise"] = "friendly",
    ) -> dict[str, Any]:
        """
        Generate narration script from content

        Args:
            content: Note content (Markdown format)
            style: Narration style

        Returns:
            Dict with script, key_points, and metadata
        """
        style_prompts = {
            "friendly": """You are a friendly and approachable tutor.
Use "we", "us", "you" to create closeness.
Relaxed but professional tone.
Use transitions like "well", "next", "so".""",
            "academic": """You are a senior scholar giving an academic lecture.
Use "we", "this paper" academic language.
Rigorous and professional tone.
Clear introduction-body-conclusion structure.""",
            "concise": """You are an efficient knowledge communicator.
Direct and to the point.
General first, then details.
Use "first", "then", "finally".""",
        }

        system_prompt = f"""You are a professional note narration script writing expert.

{style_prompts.get(style, style_prompts["friendly"])}

**Output Format**:
Output the narration script text directly, without any additional explanations.

**Notes**:
1. Maintain core information and logical structure
2. Convert Markdown to oral descriptions
3. Describe math formulas orally (e.g., "x squared plus y squared")
4. **Control length within 4000 characters**"""

        user_prompt = f"""Convert the following note content into a narration script:

---
{content[:8000]}
---

Generate a narration script suitable for oral reading (within 4000 characters)."""

        script = await self._call_llm(system_prompt, user_prompt)

        # Truncate if needed
        if len(script) > 4000:
            truncated = script[:3997]
            last_period = max(
                truncated.rfind("."),
                truncated.rfind("!"),
                truncated.rfind("?"),
            )
            script = truncated[:last_period + 1] if last_period > 3500 else truncated + "..."

        # Extract key points
        key_points = await self._extract_key_points(content)

        return {
            "script": script,
            "key_points": key_points,
            "style": style,
            "original_length": len(content),
            "script_length": len(script),
        }

    async def _extract_key_points(self, content: str) -> list[str]:
        """Extract key points from content"""
        system_prompt = """You are a content analysis expert.
Extract 3-5 key points from the given notes.

Output format: JSON array, each element is a key point string.
Example: ["Key point 1", "Key point 2", "Key point 3"]

Only output the JSON array, no other content."""

        user_prompt = f"""Extract key points from the following notes:

{content[:4000]}"""

        try:
            response = await self._call_llm(system_prompt, user_prompt, temperature=0.3)
            # Try to parse JSON from response
            import re
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        return []


# Example usage
async def main():
    """Demo of Co-Writer with Ollama"""
    cowriter = OllamaCoWriter()

    print("=" * 60)
    print("Co-Writer with Ollama Demo")
    print("=" * 60)

    # Example 1: Edit text
    print("\n1. Edit Text (Rewrite)")
    print("-" * 40)
    result = await cowriter.edit(
        text="The quick brown fox jumps over the lazy dog.",
        instruction="Make it more formal and academic",
        action="rewrite",
    )
    print(f"Original: The quick brown fox jumps over the lazy dog.")
    print(f"Edited: {result['edited_text']}")
    print(f"Operation ID: {result['operation_id']}")

    # Example 2: Auto-mark
    print("\n2. Auto-Mark Text")
    print("-" * 40)
    result = await cowriter.auto_mark(
        text="Deep learning is a subfield of machine learning. "
        "Its core is using neural networks to learn data representations. "
        "The model achieved 99.2% accuracy on the MNIST dataset."
    )
    print(f"Marked: {result['marked_text']}")

    # Example 3: Generate narration script
    print("\n3. Generate Narration Script")
    print("-" * 40)
    result = await cowriter.generate_narration_script(
        content="""# Attention Mechanism

The attention mechanism allows neural networks to focus on relevant parts of the input.

## Key Concepts
- Query, Key, Value vectors
- Scaled dot-product attention
- Multi-head attention

## Formula
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
""",
        style="friendly",
    )
    print(f"Script:\n{result['script'][:500]}...")
    print(f"\nKey Points: {result['key_points']}")
    print(f"Script Length: {result['script_length']} chars")

    print("\n" + "=" * 60)
    print("History saved to:", OUTPUT_DIR / "history.json")


if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Standalone Version

```bash
# 1. Make sure Ollama is running
ollama serve

# 2. Pull a model if needed
ollama pull llama3.2

# 3. Install dependencies
pip install openai

# 4. Run the script
python cowriter_ollama.py
```

---

## Storage Structure

```
data/user/co-writer/
├── history.json              # Edit operation history
├── audio/                    # TTS audio files (if enabled)
│   └── narration_*.mp3
└── tool_calls/               # RAG/Web search results
    └── {operation_id}_{rag|web}.json
```

### History Record Format

```json
{
  "id": "20250101_120000_abc123",
  "timestamp": "2025-01-01T12:00:00.000000",
  "action": "rewrite",
  "source": "rag",
  "kb_name": "my_kb",
  "input": {
    "original_text": "...",
    "instruction": "..."
  },
  "output": {
    "edited_text": "..."
  },
  "tool_call_file": "path/to/tool_call.json",
  "model": "llama3.2"
}
```

---

## Summary

The Co-Writer module is a well-architected system for:

1. **Text Editing**: Rewrite, shorten, or expand text with AI assistance
2. **Context Enhancement**: Optional RAG or web search for informed edits
3. **Auto-Marking**: Semantic annotation with HTML tags for reading aids
4. **Narration**: Convert text to spoken scripts with style control
5. **TTS Audio**: Generate audio files (requires OpenAI TTS or compatible)

**Key Design Patterns:**
- Uses OpenAI-compatible API (works with Ollama out of the box)
- YAML-based prompts for easy customization and i18n
- History tracking for audit and undo functionality
- Lazy-loading of optional components (TTS)
- Unified configuration via environment variables

**To duplicate with Ollama:**
1. Set `LLM_BINDING_HOST=http://localhost:11434/v1`
2. Set `LLM_MODEL` to your Ollama model name
3. Use the minimal implementation above or integrate with full DeepTutor
