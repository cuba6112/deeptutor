#!/usr/bin/env python
"""
Ollama Web Search Tool - Web search using Ollama's built-in search API

This provides an alternative to Perplexity for web research capabilities.
Requires an Ollama API key from https://ollama.com/settings/keys
"""

from datetime import datetime
import json
import os
from typing import Any

# Check for ollama module
try:
    from ollama import web_search as ollama_search
    from ollama import web_fetch as ollama_fetch
    OLLAMA_WEB_AVAILABLE = True
except ImportError:
    OLLAMA_WEB_AVAILABLE = False
    ollama_search = None
    ollama_fetch = None


def ollama_web_search(
    query: str,
    max_results: int = 5,
    output_dir: str | None = None,
    verbose: bool = False
) -> dict[str, Any]:
    """
    Perform web search using Ollama's built-in search API.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
        output_dir: Output directory (optional, if provided will save results)
        verbose: Whether to print detailed information

    Returns:
        dict: Dictionary containing search results
            {
                "query": str,
                "answer": str (summarized from results),
                "search_results": list[dict],
                "citations": list[dict],
                "result_file": str (if file was saved)
            }

    Raises:
        ImportError: If ollama module doesn't have web_search
        ValueError: If OLLAMA_API_KEY environment variable is not set
        Exception: If API call fails
    """
    if not OLLAMA_WEB_AVAILABLE:
        raise ImportError(
            "Ollama web search is not available. Please update ollama package:\n"
            "  pip install --upgrade ollama\n"
            "Note: Web search requires ollama >= 0.4.0"
        )

    # Check API key
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY environment variable is not set.\n"
            "Get your API key from: https://ollama.com/settings/keys"
        )

    try:
        # Call Ollama web search API
        raw_results = ollama_search(query, max_results=max_results)

        # Build standardized result format (compatible with existing web_search interface)
        search_results = []
        citations = []

        if raw_results and "results" in raw_results:
            for i, item in enumerate(raw_results["results"], 1):
                search_result = {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "snippet": item.get("content", "")[:300] if item.get("content") else "",
                    "source": "ollama_web_search",
                    "date": None,
                }
                search_results.append(search_result)

                # Build citation
                citations.append({
                    "id": i,
                    "reference": f"[{i}]",
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:200] if item.get("content") else "",
                })

        # Create a summary answer from the search results
        answer_parts = []
        for result in search_results[:3]:  # Use top 3 results for summary
            if result["content"]:
                answer_parts.append(result["content"][:500])

        answer = "\n\n".join(answer_parts) if answer_parts else "No results found."

        result = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "model": "ollama_web_search",
            "answer": answer,
            "citations": citations,
            "search_results": search_results,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        # If output directory provided, save results
        result_file = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"ollama_search_{timestamp}.json"
            output_path = os.path.join(output_dir, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            result_file = output_path

            if verbose:
                print(f"Search results saved to: {output_path}")

        if result_file:
            result["result_file"] = result_file

        if verbose:
            print(f"Query: {query}")
            print(f"Found {len(search_results)} results")
            if answer:
                print(f"Answer preview: {answer[:200]}...")

        return result

    except Exception as e:
        raise Exception(f"Ollama web search failed: {e!s}")


def ollama_web_fetch(
    url: str,
    verbose: bool = False
) -> dict[str, Any]:
    """
    Fetch content from a specific URL using Ollama's web fetch API.

    Args:
        url: URL to fetch content from
        verbose: Whether to print detailed information

    Returns:
        dict: Dictionary containing fetched content
            {
                "url": str,
                "title": str,
                "content": str,
                "links": list[str]
            }

    Raises:
        ImportError: If ollama module doesn't have web_fetch
        ValueError: If OLLAMA_API_KEY is not set
        Exception: If fetch fails
    """
    if not OLLAMA_WEB_AVAILABLE:
        raise ImportError(
            "Ollama web fetch is not available. Please update ollama package:\n"
            "  pip install --upgrade ollama"
        )

    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError(
            "OLLAMA_API_KEY environment variable is not set.\n"
            "Get your API key from: https://ollama.com/settings/keys"
        )

    try:
        result = ollama_fetch(url)

        fetched = {
            "url": url,
            "title": getattr(result, "title", ""),
            "content": getattr(result, "content", ""),
            "links": getattr(result, "links", []),
        }

        if verbose:
            print(f"Fetched: {url}")
            print(f"Title: {fetched['title']}")
            print(f"Content length: {len(fetched['content'])} chars")

        return fetched

    except Exception as e:
        raise Exception(f"Ollama web fetch failed: {e!s}")


# Unified web search function that can use either Ollama or Perplexity
def web_search_unified(
    query: str,
    provider: str = "auto",
    max_results: int = 5,
    output_dir: str | None = None,
    verbose: bool = False
) -> dict[str, Any]:
    """
    Unified web search that automatically selects the best available provider.

    Args:
        query: Search query string
        provider: "ollama", "perplexity", or "auto" (default)
        max_results: Maximum number of results
        output_dir: Output directory for saving results
        verbose: Whether to print detailed information

    Returns:
        dict: Search results in standardized format
    """
    if provider == "auto":
        # Try Ollama first (free), then Perplexity
        if OLLAMA_WEB_AVAILABLE and os.environ.get("OLLAMA_API_KEY"):
            provider = "ollama"
        elif os.environ.get("PERPLEXITY_API_KEY"):
            provider = "perplexity"
        else:
            raise ValueError(
                "No web search provider configured. Set one of:\n"
                "  - OLLAMA_API_KEY (from https://ollama.com/settings/keys)\n"
                "  - PERPLEXITY_API_KEY"
            )

    if provider == "ollama":
        return ollama_web_search(query, max_results, output_dir, verbose)
    elif provider == "perplexity":
        # Import and use Perplexity
        from src.tools.web_search import web_search as perplexity_search
        return perplexity_search(query, output_dir, verbose)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'ollama', 'perplexity', or 'auto'")


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # Test
    try:
        result = ollama_web_search(
            "What are the latest developments in AI?",
            max_results=3,
            verbose=True
        )
        print("\nSearch completed!")
        print(f"Query: {result['query']}")
        print(f"Results: {len(result['search_results'])}")
        for r in result['search_results']:
            print(f"  - {r['title']}: {r['url']}")
    except Exception as e:
        print(f"Error: {e}")
