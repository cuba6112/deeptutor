"""
Tests for JSON parsing utilities.

Tests the extract_json_from_response and parse_json_response functions
that handle markdown-wrapped JSON responses from LLMs (especially Ollama models).
"""

import json

import pytest

from src.utils.json_utils import (
    extract_json_from_response,
    parse_json_response,
    safe_parse_json_response,
)


class TestExtractJsonFromResponse:
    """Tests for extract_json_from_response function."""

    def test_raw_json_object(self):
        """Test extraction of raw JSON object without markdown."""
        response = '{"key": "value", "number": 42}'
        result = extract_json_from_response(response)
        assert result == '{"key": "value", "number": 42}'

    def test_raw_json_array(self):
        """Test extraction of raw JSON array without markdown."""
        response = '[{"item": 1}, {"item": 2}]'
        result = extract_json_from_response(response)
        assert result == '[{"item": 1}, {"item": 2}]'

    def test_markdown_json_block(self):
        """Test extraction from ```json ... ``` code block."""
        response = '```json\n{"key": "value", "number": 42}\n```'
        result = extract_json_from_response(response)
        assert json.loads(result) == {"key": "value", "number": 42}

    def test_markdown_block_no_language(self):
        """Test extraction from ``` ... ``` code block without language."""
        response = '```\n{"key": "value", "number": 42}\n```'
        result = extract_json_from_response(response)
        assert json.loads(result) == {"key": "value", "number": 42}

    def test_markdown_with_surrounding_text(self):
        """Test extraction when markdown block has surrounding text."""
        response = 'Here is the JSON:\n```json\n{"key": "value"}\n```\nEnd of response.'
        result = extract_json_from_response(response)
        assert json.loads(result) == {"key": "value"}

    def test_nested_json(self):
        """Test extraction of nested JSON objects."""
        nested = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        response = f'```json\n{json.dumps(nested)}\n```'
        result = extract_json_from_response(response)
        assert json.loads(result) == nested

    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        response = '  \n  {"key": "value"}  \n  '
        result = extract_json_from_response(response)
        assert json.loads(result) == {"key": "value"}

    def test_markdown_with_whitespace(self):
        """Test markdown block with extra whitespace."""
        response = '  \n```json\n  {"key": "value"}  \n```\n  '
        result = extract_json_from_response(response)
        assert json.loads(result) == {"key": "value"}

    def test_empty_response(self):
        """Test handling of empty response."""
        assert extract_json_from_response("") == ""
        assert extract_json_from_response(None) == ""

    def test_case_insensitive_json_tag(self):
        """Test that ```JSON and ```json both work."""
        response = '```JSON\n{"key": "value"}\n```'
        result = extract_json_from_response(response)
        assert json.loads(result) == {"key": "value"}


class TestParseJsonResponse:
    """Tests for parse_json_response function."""

    def test_parse_raw_json(self):
        """Test parsing raw JSON."""
        response = '{"key": "value", "number": 42}'
        result = parse_json_response(response)
        assert result == {"key": "value", "number": 42}

    def test_parse_markdown_wrapped_json(self):
        """Test parsing markdown-wrapped JSON."""
        response = '```json\n{"key": "value", "number": 42}\n```'
        result = parse_json_response(response)
        assert result == {"key": "value", "number": 42}

    def test_parse_json_array(self):
        """Test parsing JSON array."""
        response = '```json\n[{"item": 1}, {"item": 2}]\n```'
        result = parse_json_response(response)
        assert result == [{"item": 1}, {"item": 2}]

    def test_parse_nested_json(self):
        """Test parsing nested JSON."""
        nested = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        response = f'```json\n{json.dumps(nested)}\n```'
        result = parse_json_response(response)
        assert result == nested

    def test_parse_empty_response_raises(self):
        """Test that empty response raises ValueError."""
        with pytest.raises(ValueError, match="Empty response"):
            parse_json_response("")

    def test_parse_invalid_json_raises(self):
        """Test that invalid JSON raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("not valid json")

    def test_parse_only_text_raises(self):
        """Test that text-only response without JSON raises error."""
        with pytest.raises(json.JSONDecodeError):
            parse_json_response("This is just some text without JSON")


class TestSafeParseJsonResponse:
    """Tests for safe_parse_json_response function."""

    def test_successful_parse(self):
        """Test successful parsing returns result and no error."""
        response = '```json\n{"key": "value"}\n```'
        result, error = safe_parse_json_response(response)
        assert result == {"key": "value"}
        assert error is None

    def test_failed_parse_returns_default(self):
        """Test failed parsing returns default value."""
        response = "not valid json"
        result, error = safe_parse_json_response(response, default={"default": True})
        assert result == {"default": True}
        assert error is not None
        assert "JSON parsing failed" in error

    def test_empty_response_returns_default(self):
        """Test empty response returns default value."""
        result, error = safe_parse_json_response("", default=[])
        assert result == []
        assert error is not None

    def test_default_is_none(self):
        """Test default value is None when not specified."""
        result, error = safe_parse_json_response("invalid")
        assert result is None
        assert error is not None


class TestOllamaModelResponses:
    """
    Tests specifically for Ollama model response formats.

    Ollama models (like minimax-m2.1:cloud) often wrap JSON in markdown blocks.
    These tests ensure we handle all common Ollama response patterns.
    """

    def test_ollama_typical_response(self):
        """Test typical Ollama JSON response format."""
        # This is the format that was causing "JSON parsing failed: Expecting value" errors
        response = '''```json
{
    "queries": ["attention mechanism", "transformer architecture"],
    "status": "success"
}
```'''
        result = parse_json_response(response)
        assert result["queries"] == ["attention mechanism", "transformer architecture"]
        assert result["status"] == "success"

    def test_ollama_with_explanation(self):
        """Test Ollama response with explanation text before/after JSON."""
        response = '''I'll provide the JSON response:

```json
{
    "relevant": true,
    "reason": "The query matches the knowledge base content"
}
```

This indicates the content is relevant.'''
        result = parse_json_response(response)
        assert result["relevant"] is True
        assert "reason" in result

    def test_ollama_nested_response(self):
        """Test Ollama response with nested JSON structure."""
        response = '''```json
{
    "focuses": [
        {"id": "q_1", "focus": "Basic concept", "scenario_hint": "Example 1"},
        {"id": "q_2", "focus": "Advanced topic", "scenario_hint": "Example 2"}
    ],
    "total": 2
}
```'''
        result = parse_json_response(response)
        assert len(result["focuses"]) == 2
        assert result["focuses"][0]["id"] == "q_1"

    def test_ollama_unicode_content(self):
        """Test Ollama response with Unicode content (e.g., Chinese)."""
        response = '''```json
{
    "knowledge_point": "注意力机制",
    "difficulty": "medium",
    "question_type": "written"
}
```'''
        result = parse_json_response(response)
        assert result["knowledge_point"] == "注意力机制"
        assert result["difficulty"] == "medium"

    def test_ollama_array_response(self):
        """Test Ollama response that returns a JSON array."""
        response = '''```json
[
    {"title": "Question 1", "content": "What is..."},
    {"title": "Question 2", "content": "Explain..."}
]
```'''
        result = parse_json_response(response)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Question 1"


class TestEdgeCases:
    """Tests for edge cases and unusual input formats."""

    def test_multiple_json_blocks(self):
        """Test response with multiple JSON blocks - should extract first valid one."""
        response = '''First block:
```json
{"first": true}
```
Second block:
```json
{"second": true}
```'''
        result = parse_json_response(response)
        # Should extract the first valid JSON block
        assert "first" in result or "second" in result

    def test_escaped_characters(self):
        """Test JSON with escaped characters."""
        response = '```json\n{"message": "Hello \\"world\\""}\n```'
        result = parse_json_response(response)
        assert result["message"] == 'Hello "world"'

    def test_newlines_in_values(self):
        """Test JSON with newlines in string values."""
        response = '```json\n{"content": "Line 1\\nLine 2\\nLine 3"}\n```'
        result = parse_json_response(response)
        assert "Line 1\nLine 2\nLine 3" == result["content"]

    def test_large_numbers(self):
        """Test JSON with large numbers."""
        response = '```json\n{"large": 9999999999999999}\n```'
        result = parse_json_response(response)
        assert result["large"] == 9999999999999999

    def test_boolean_values(self):
        """Test JSON with boolean values."""
        response = '```json\n{"active": true, "deleted": false}\n```'
        result = parse_json_response(response)
        assert result["active"] is True
        assert result["deleted"] is False

    def test_null_values(self):
        """Test JSON with null values."""
        response = '```json\n{"value": null}\n```'
        result = parse_json_response(response)
        assert result["value"] is None
