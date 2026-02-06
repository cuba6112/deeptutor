"""
Pytest configuration and shared fixtures for DeepTutor tests.
"""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def project_root_path():
    """Return the project root path."""
    return project_root


@pytest.fixture
def sample_json_responses():
    """Sample LLM responses in various formats for testing JSON extraction."""
    return {
        "raw_json": '{"key": "value", "number": 42}',
        "markdown_json": '```json\n{"key": "value", "number": 42}\n```',
        "markdown_no_lang": '```\n{"key": "value", "number": 42}\n```',
        "json_with_text": 'Here is the result:\n```json\n{"key": "value", "number": 42}\n```\nLet me know if you need more.',
        "array_json": '[{"item": 1}, {"item": 2}]',
        "markdown_array": '```json\n[{"item": 1}, {"item": 2}]\n```',
        "nested_json": '{"outer": {"inner": "value"}, "list": [1, 2, 3]}',
        "markdown_nested": '```json\n{"outer": {"inner": "value"}, "list": [1, 2, 3]}\n```',
        "whitespace_json": '  \n  {"key": "value"}  \n  ',
        "markdown_whitespace": '  \n```json\n  {"key": "value"}  \n```\n  ',
    }


@pytest.fixture
def sample_rag_query():
    """Sample RAG query for testing."""
    return "What is the attention mechanism in transformers?"


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration for testing."""
    return {
        "binding": "openai",
        "model": "test-model",
        "api_key": "test-api-key",
        "base_url": "http://localhost:11434/v1",
    }


@pytest.fixture
def mock_embedding_config():
    """Mock embedding configuration for testing."""
    return {
        "binding": "openai",
        "model": "test-embedding-model",
        "api_key": "test-api-key",
        "base_url": "http://localhost:11434/v1",
        "dim": 768,
        "max_tokens": 8192,
    }


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, tmp_path):
    """Set up test environment variables."""
    # Set required environment variables for testing
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BINDING_HOST", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "test-embedding")
    monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://localhost:11434/v1")
    monkeypatch.setenv("EMBEDDING_DIM", "768")
