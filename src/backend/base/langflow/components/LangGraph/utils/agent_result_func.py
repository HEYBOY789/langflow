import json  # noqa: INP001
import re
from typing import Any


def clarify_result(result):
    """Clarify the result from the agent or crew response."""
    # If pydantic model is in result, get it and convert it to dict
    if hasattr(result, "pydantic"):
        response_data = result.pydantic
        if hasattr(response_data, "model_dump"):
            print("Response has a Pydantic result, converting to dict")  # noqa: T201
            return response_data.model_dump()

    # Result is normal string
    print("Parsing result as text")  # noqa: T201
    return (
        parse_json_from_text(str(result.raw))
        if hasattr(result, "raw")
        else parse_json_from_text(str(result))
    )

def parse_json_from_text(text: str) -> dict[str, Any]:
    """Extract and parse JSON from text that might contain additional content."""
    print("Debugging text parsing:", text)  # noqa: T201
    try:
        # First, try parsing the entire text as JSON
        return safe_literal_eval(text, dict)
    except (ValueError, SyntaxError, TypeError):
        # If that fails, try to extract JSON from the text
        # Look for JSON-like structures
        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, text, re.DOTALL)
        print(f"Found {len(matches)} potential JSON matches in the text.")  # noqa: T201

        for match in matches:
            try:
                # Clean up the JSON string
                cleaned_json = clean_json_string(match)
                print(f"Trying to parse cleaned JSON: {cleaned_json}")  # noqa: T201
                return json.loads(cleaned_json)
            except (json.JSONDecodeError, ValueError):
                return safe_literal_eval(cleaned_json, dict)

        # If no valid JSON found, raise an error
        msg = "Could not extract valid JSON from response..."
        raise ValueError(msg) from None

def clean_json_string(json_str: str) -> str:
    """Clean up common JSON formatting issues."""
    # Remove control characters and fix common issues
    json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", json_str)
    # Fix trailing commas
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    return json_str.strip()

def safe_literal_eval(value: str, expected_type=None):
    """Safely evaluate literal with type checking."""
    import ast
    result = ast.literal_eval(value)
    if expected_type and not isinstance(result, expected_type):
        msg = f"Expected {expected_type.__name__}, got {type(result).__name__}"
        raise ValueError(msg)
    return result
