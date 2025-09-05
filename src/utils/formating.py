import json
import logging
import re
from typing import Any, Dict, List, Union

import nltk


def extract_json(
    response: str,
    fields: List[str] = ["búsquedas", "preguntas"],
) -> Union[Dict[str, List[str]], None]:
    """
    Parse the web search answers to extract the specified fields.

    Args:
        response (str): The response to parse, potentially containing JSON.
        fields (List[str]): The fields to extract from the JSON.

    Returns:
        Dict[str, List[str]]: The parsed fields, or None if not found.
    """
    # Preprocess the response
    response = preprocess_json(response)

    # Extract JSON data
    data = extract_first_json(response)
    data = [d for d in data if isinstance(d, dict)]

    if not data:
        logging.warning(f"No valid JSON found in the response: {response}")
        return None

    # Use the largest JSON object if multiple were found
    data = max(data, key=lambda x: len(json.dumps(x)))

    # Check if the JSON has the required fields
    field_dict = {}
    for field in fields:
        field_dict[field] = None
        for key in data.keys():
            key_lower = key.lower()
            if field.lower() in key_lower or nltk.edit_distance(field.lower(), key_lower) <= 2:
                field_dict[field] = data[key]
                break

    # If all fields are found, return them
    if all(field_dict.values()):
        return field_dict

    # If some fields are missing, try to extract them from the text
    for field in fields:
        if field_dict[field] is None:
            pattern = rf'{field}["\':\s]*(\[.*?\])'
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    field_dict[field] = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

    if all(field_dict.values()):
        return field_dict
    else:
        logging.warning(f"Couldn't find all required fields in the response: {response}")
        return None


def preprocess_json(text: str) -> str:
    """
    Attempt to fix common JSON formatting issues, such as trailing commas.
    """

    if text.startswith("```json"):
        text = text.replace("```json", "")
        if text.endswith("```"):
            text = text.replace("```", "")
        text = text.strip()

    # This regex aims to find trailing commas before closing brackets or braces and remove them.
    # Note: This is a simple approach and may not work correctly in all cases, e.g., when commas are within strings.
    text = re.sub(r",\s*([\]}])", r"\1", text).strip()
    return text


def extract_first_json(text: str) -> List[Dict[str, Any]]:
    """
    Extract the first JSON object or array from unstructured text.

    Arg:
        text (str): The text to search for JSON.

    Returns:
        List[Dict[str,Any]]: The parsed JSON object or array, or None if no valid JSON found.
    """

    try:
        # Attempt to parse the entire text as JSON
        parsed_json = json.loads(text)
        return [parsed_json]
    except json.JSONDecodeError:
        pass

    try:
        # Attempt to parse the entire text as JSON after preprocessing
        parsed_json = json.loads(preprocess_json(text))
        return [parsed_json]
    except json.JSONDecodeError:
        pass

    # Regular expression to find JSON objects or arrays
    # Function to match curly and square brackets, accounting for nesting

    def match_brackets(substr, open_bracket, close_bracket):
        stack = []
        for i, char in enumerate(substr):
            if char == open_bracket:
                stack.append(i)
            elif char == close_bracket and stack:
                start = stack.pop()
                if not stack:
                    return substr[: i + 1]
        return None

    # Find all possible JSON starts
    potential_starts = [match.start() for match in re.finditer(r"[\{\[]", text)]
    # print(potential_starts)
    results = []
    for start in potential_starts:
        # Check if the segment can be a valid JSON
        segment = text[start:]
        if segment.startswith("{"):
            matched = match_brackets(segment, "{", "}")
        elif segment.startswith("["):
            matched = match_brackets(segment, "[", "]")
        else:
            continue  # Not a valid start

        if matched:
            try:
                # Attempt to parse the matched segment as JSON
                parsed_json = json.loads(matched)
                results.append(parsed_json)
                # Skip past this segment for the next iteration
                start += len(matched)
            except json.JSONDecodeError:
                continue
                # Optionally, continue to try other matches or break

    # Return all found and parsed JSON objects/arrays
    return results
