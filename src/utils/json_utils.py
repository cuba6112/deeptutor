"""
JSON Utilities - Helper functions for parsing JSON responses from LLMs.

Handles cases where LLMs return JSON wrapped in markdown code blocks.
"""

import json
import re
from typing import Any


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON from LLM response that may be wrapped in markdown code blocks.

    Handles formats like:
    - Raw JSON: {"key": "value"}
    - Markdown wrapped: ```json\n{"key": "value"}\n```
    - Markdown without language: ```\n{"key": "value"}\n```

    Args:
        response: Raw LLM response string

    Returns:
        Extracted JSON string (without markdown wrapper)
    """
    if not response:
        return ""

    # Strip whitespace
    response = response.strip()

    # Try to extract from markdown code blocks
    # Pattern matches ```json ... ``` or ``` ... ```
    patterns = [
        r"```json\s*\n?(.*?)\n?```",  # ```json ... ```
        r"```\s*\n?(.*?)\n?```",       # ``` ... ```
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            # Verify it looks like JSON
            if extracted.startswith(("{", "[")):
                return extracted

    # If no code block found, return original (might already be raw JSON)
    return response


def parse_json_response(response: str) -> Any:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON object (dict or list)

    Raises:
        json.JSONDecodeError: If JSON parsing fails
        ValueError: If response is empty
    """
    if not response or not response.strip():
        raise ValueError("Empty response cannot be parsed as JSON")

    json_str = extract_json_from_response(response)

    if not json_str:
        raise ValueError("No JSON content found in response")

    return json.loads(json_str)


def safe_parse_json_response(response: str, default: Any = None) -> tuple[Any, str | None]:
    """
    Safely parse JSON from LLM response, returning default on failure.

    Args:
        response: Raw LLM response string
        default: Default value to return on failure

    Returns:
        Tuple of (parsed_result, error_message)
        - On success: (parsed_json, None)
        - On failure: (default, error_message)
    """
    try:
        result = parse_json_response(response)
        return result, None
    except json.JSONDecodeError as e:
        return default, f"JSON parsing failed: {e!s}"
    except ValueError as e:
        return default, str(e)
    except Exception as e:
        return default, f"Unexpected error: {e!s}"
