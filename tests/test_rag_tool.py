"""
Tests for RAG (Retrieval-Augmented Generation) tool.

Tests the rag_search function and related RAG functionality.
Note: Some tests require actual knowledge bases and may be skipped in CI.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRagToolImports:
    """Tests for RAG tool imports and module availability."""

    def test_import_rag_tool(self):
        """Test that rag_tool module can be imported."""
        from src.tools import rag_tool

        assert hasattr(rag_tool, "rag_search")

    def test_import_rag_search_function(self):
        """Test that rag_search function can be imported directly."""
        from src.tools.rag_tool import rag_search

        assert callable(rag_search)


class TestRagSearchParameters:
    """Tests for rag_search function parameter handling."""

    def test_valid_query_modes(self):
        """Test that valid query modes are accepted."""
        # This tests the function signature, not actual execution
        from src.tools.rag_tool import rag_search

        # Check function accepts the expected parameters
        import inspect

        sig = inspect.signature(rag_search)
        params = sig.parameters

        assert "query" in params
        assert "kb_name" in params
        assert "mode" in params
        assert "api_key" in params
        assert "base_url" in params
        assert "kb_base_dir" in params

    def test_default_mode_is_hybrid(self):
        """Test that default mode parameter is 'hybrid'."""
        import inspect

        from src.tools.rag_tool import rag_search

        sig = inspect.signature(rag_search)
        mode_param = sig.parameters["mode"]
        assert mode_param.default == "hybrid"


class TestKnowledgeBaseManager:
    """Tests for KnowledgeBaseManager class."""

    def test_manager_initialization(self, tmp_path):
        """Test KnowledgeBaseManager initialization."""
        from src.knowledge.manager import KnowledgeBaseManager

        manager = KnowledgeBaseManager(str(tmp_path))
        assert manager is not None

    def test_list_empty_knowledge_bases(self, tmp_path):
        """Test listing knowledge bases when none exist."""
        from src.knowledge.manager import KnowledgeBaseManager

        manager = KnowledgeBaseManager(str(tmp_path))
        kbs = manager.list_knowledge_bases()
        assert isinstance(kbs, list)

    def test_get_default_kb_when_none_set(self, tmp_path):
        """Test getting default KB when none is set."""
        from src.knowledge.manager import KnowledgeBaseManager

        manager = KnowledgeBaseManager(str(tmp_path))
        default = manager.get_default()
        # Should return None or raise appropriate error
        assert default is None or isinstance(default, str)


class TestRagSearchMocked:
    """Tests for rag_search with mocked dependencies."""

    @pytest.fixture
    def mock_rag_anything(self):
        """Create a mock RAGAnything instance."""
        mock_rag = MagicMock()
        mock_rag.aquery = AsyncMock(return_value="Mocked answer from RAG")
        mock_rag._ensure_lightrag_initialized = AsyncMock()
        return mock_rag

    @pytest.mark.asyncio
    async def test_rag_search_returns_dict(self, mock_rag_anything, tmp_path, monkeypatch):
        """Test that rag_search returns a dictionary with expected keys."""
        # Create a minimal KB structure
        kb_path = tmp_path / "test_kb" / "rag_storage"
        kb_path.mkdir(parents=True)

        # Create required files for KB manager
        config_path = tmp_path / "kb_config.json"
        config_path.write_text('{"default_kb": "test_kb", "knowledge_bases": ["test_kb"]}')

        metadata_path = tmp_path / "test_kb" / "metadata.json"
        metadata_path.write_text('{"name": "test_kb"}')

        with patch("src.tools.rag_tool.RAGAnything", return_value=mock_rag_anything):
            from src.tools.rag_tool import rag_search

            result = await rag_search(
                query="Test query",
                kb_name="test_kb",
                mode="naive",
                kb_base_dir=str(tmp_path),
            )

            assert isinstance(result, dict)
            assert "query" in result
            assert "answer" in result
            assert "mode" in result

    @pytest.mark.asyncio
    async def test_rag_search_query_modes(self, mock_rag_anything, tmp_path):
        """Test that different query modes work."""
        # Create KB structure
        kb_path = tmp_path / "test_kb" / "rag_storage"
        kb_path.mkdir(parents=True)

        config_path = tmp_path / "kb_config.json"
        config_path.write_text('{"default_kb": "test_kb", "knowledge_bases": ["test_kb"]}')

        metadata_path = tmp_path / "test_kb" / "metadata.json"
        metadata_path.write_text('{"name": "test_kb"}')

        modes = ["naive", "local", "global", "hybrid"]

        with patch("src.tools.rag_tool.RAGAnything", return_value=mock_rag_anything):
            from src.tools.rag_tool import rag_search

            for mode in modes:
                result = await rag_search(
                    query="Test query",
                    kb_name="test_kb",
                    mode=mode,
                    kb_base_dir=str(tmp_path),
                )

                assert result["mode"] == mode


class TestQueryItemTool:
    """Tests for the query_item_tool functionality."""

    def test_import_query_item_tool(self):
        """Test that query_item_tool can be imported."""
        from src.tools.query_item_tool import query_numbered_item

        assert callable(query_numbered_item)

    def test_query_item_returns_dict(self, tmp_path):
        """Test that query_numbered_item returns expected structure."""
        from src.tools.query_item_tool import query_numbered_item

        # Create a minimal KB structure with numbered_items.json
        kb_path = tmp_path / "test_kb"
        kb_path.mkdir(parents=True)

        # numbered_items.json expects a dict format, not a list
        numbered_items_path = kb_path / "numbered_items.json"
        numbered_items_path.write_text('{"Definition 1.1": {"text": "Test definition content"}}')

        config_path = tmp_path / "kb_config.json"
        config_path.write_text('{"default_kb": "test_kb", "knowledge_bases": ["test_kb"]}')

        result = query_numbered_item(
            identifier="Definition 1.1",
            kb_name="test_kb",
            kb_base_dir=str(tmp_path),
        )

        assert isinstance(result, dict)
        assert "status" in result


class TestOllamaWebSearch:
    """Tests for Ollama web search tool."""

    def test_import_ollama_web_search(self):
        """Test that ollama_web_search module can be imported."""
        from src.tools.ollama_web_search import (
            OLLAMA_WEB_AVAILABLE,
            ollama_web_search,
            web_search_unified,
        )

        assert callable(ollama_web_search)
        assert callable(web_search_unified)
        assert isinstance(OLLAMA_WEB_AVAILABLE, bool)

    def test_web_search_unified_requires_config(self, monkeypatch):
        """Test that web_search_unified raises error without config."""
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)

        from src.tools.ollama_web_search import web_search_unified

        with pytest.raises(ValueError, match="No web search provider configured"):
            web_search_unified("test query", provider="auto")


class TestRagIntegration:
    """Integration tests for RAG system (may require actual setup)."""

    @pytest.mark.skip(reason="Requires actual knowledge base and LLM")
    @pytest.mark.asyncio
    async def test_full_rag_query(self):
        """Test full RAG query with actual knowledge base."""
        from src.tools.rag_tool import rag_search

        result = await rag_search(
            query="What is attention mechanism?",
            kb_name="test_kb",
            mode="hybrid",
        )

        assert result["answer"]
        assert len(result["answer"]) > 0

    @pytest.mark.skip(reason="Requires actual Ollama server")
    @pytest.mark.asyncio
    async def test_rag_with_ollama(self):
        """Test RAG with actual Ollama model."""
        from src.tools.rag_tool import rag_search

        result = await rag_search(
            query="Explain the concept",
            kb_name="test_kb",
            mode="naive",
        )

        assert "answer" in result
