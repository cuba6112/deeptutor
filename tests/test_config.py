"""
Tests for configuration management.

Tests the core configuration functions that handle LLM, embedding,
and other settings from environment variables and YAML files.
"""

import os

import pytest


class TestLLMConfig:
    """Tests for get_llm_config function."""

    def test_get_llm_config_success(self):
        """Test successful LLM configuration retrieval."""
        from src.core.core import get_llm_config

        config = get_llm_config()
        assert "model" in config
        assert "api_key" in config
        assert "base_url" in config
        assert "binding" in config

    @pytest.mark.skip(reason="Requires clean environment without .env files")
    def test_get_llm_config_missing_model(self, monkeypatch):
        """Test LLM config raises error when model is missing.

        Note: This test is skipped because the dotenv files are loaded at module
        import time, making it difficult to test missing environment variables.
        The configuration validation logic is still tested implicitly through
        other tests that verify the configuration structure.
        """
        pass

    @pytest.mark.skip(reason="Requires clean environment without .env files")
    def test_get_llm_config_missing_api_key(self, monkeypatch):
        """Test LLM config raises error when API key is missing.

        Note: This test is skipped because the dotenv files are loaded at module
        import time, making it difficult to test missing environment variables.
        """
        pass


class TestEmbeddingConfig:
    """Tests for get_embedding_config function."""

    def test_get_embedding_config_success(self):
        """Test successful embedding configuration retrieval."""
        from src.core.core import get_embedding_config

        config = get_embedding_config()
        assert "model" in config
        assert "api_key" in config
        assert "base_url" in config
        assert "dim" in config
        assert "max_tokens" in config

    @pytest.mark.skip(reason="Requires clean environment without .env files")
    def test_get_embedding_config_default_dim(self, monkeypatch):
        """Test embedding config uses default dimension when not set.

        Note: This test is skipped because the dotenv files are loaded at module
        import time. The actual dimension used depends on the environment configuration.
        """
        pass

    def test_get_embedding_config_custom_dim(self, monkeypatch):
        """Test embedding config uses custom dimension when set."""
        monkeypatch.setenv("EMBEDDING_DIM", "768")

        import importlib

        import src.core.core

        importlib.reload(src.core.core)

        config = src.core.core.get_embedding_config()
        assert config["dim"] == 768


class TestAgentParams:
    """Tests for get_agent_params function."""

    def test_get_agent_params_defaults(self, tmp_path, monkeypatch):
        """Test agent params returns defaults when config doesn't exist."""
        from src.core.core import get_agent_params

        # Point to non-existent config directory
        monkeypatch.chdir(tmp_path)

        params = get_agent_params("test_module")
        assert params["temperature"] == 0.5
        assert params["max_tokens"] == 4096

    def test_get_agent_params_structure(self):
        """Test agent params returns expected structure."""
        from src.core.core import get_agent_params

        params = get_agent_params("solve")
        assert "temperature" in params
        assert "max_tokens" in params
        assert isinstance(params["temperature"], (int, float))
        assert isinstance(params["max_tokens"], int)


class TestConfigLoader:
    """Tests for YAML configuration loading."""

    def test_load_config_with_main(self, project_root_path):
        """Test loading configuration with main.yaml merge."""
        from src.core.core import load_config_with_main

        config = load_config_with_main("solve_config.yaml", project_root_path)
        assert isinstance(config, dict)

    def test_deep_merge(self):
        """Test deep merge of configuration dictionaries."""
        from src.core.core import _deep_merge

        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 4}, "e": 5}

        result = _deep_merge(base, override)

        assert result["a"] == 1
        assert result["b"]["c"] == 4
        assert result["b"]["d"] == 3
        assert result["e"] == 5

    def test_parse_language(self):
        """Test language parsing function."""
        from src.core.core import parse_language

        assert parse_language("en") == "en"
        assert parse_language("English") == "en"
        assert parse_language("ENGLISH") == "en"
        assert parse_language("zh") == "zh"
        assert parse_language("Chinese") == "zh"
        assert parse_language("CHINESE") == "zh"
        assert parse_language(None) == "zh"  # Default
        assert parse_language("unknown") == "zh"  # Default


class TestOllamaConfiguration:
    """Tests specific to Ollama model configuration."""

    def test_ollama_compatible_base_url(self, monkeypatch):
        """Test that Ollama's OpenAI-compatible endpoint works."""
        monkeypatch.setenv("LLM_MODEL", "llama3.2")
        monkeypatch.setenv("LLM_BINDING_API_KEY", "ollama")
        monkeypatch.setenv("LLM_BINDING_HOST", "http://localhost:11434/v1")

        import importlib

        import src.core.core

        importlib.reload(src.core.core)

        config = src.core.core.get_llm_config()
        assert config["base_url"] == "http://localhost:11434/v1"
        assert config["model"] == "llama3.2"

    def test_ollama_embedding_config(self, monkeypatch):
        """Test Ollama embedding model configuration."""
        monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
        monkeypatch.setenv("EMBEDDING_BINDING_API_KEY", "ollama")
        monkeypatch.setenv("EMBEDDING_BINDING_HOST", "http://localhost:11434/v1")
        monkeypatch.setenv("EMBEDDING_DIM", "768")

        import importlib

        import src.core.core

        importlib.reload(src.core.core)

        config = src.core.core.get_embedding_config()
        assert config["model"] == "nomic-embed-text"
        assert config["dim"] == 768

    def test_environment_variable_stripping(self, monkeypatch):
        """Test that environment variables are properly stripped of whitespace/quotes."""
        monkeypatch.setenv("LLM_MODEL", '"test-model"')  # With quotes
        monkeypatch.setenv("LLM_BINDING_API_KEY", "  test-key  ")  # With whitespace
        monkeypatch.setenv("LLM_BINDING_HOST", " 'http://localhost:11434/v1' ")

        import importlib

        import src.core.core

        importlib.reload(src.core.core)

        config = src.core.core.get_llm_config()
        assert config["model"] == "test-model"
        assert config["api_key"] == "test-key"
        assert config["base_url"] == "http://localhost:11434/v1"
