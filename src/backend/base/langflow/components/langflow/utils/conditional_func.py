import re  # noqa: INP001
from collections.abc import Callable

from langgraph.graph import START


def evaluate_condition(input_text: str | dict | list, match_text: str, operator: str, *, case_sensitive: bool) -> bool:
    print(f"Evaluating condition: {input_text} {operator} {match_text} (case_sensitive={case_sensitive})")  # noqa: T201

    # Handle List operations
    if isinstance(input_text, list):
        if operator == "contains":
            # Check if match_text is in the list
            if case_sensitive:
                return match_text in input_text
            return any(str(item).lower() == match_text.lower() for item in input_text)

        if operator == "not contains":
            # Check if match_text is NOT in the list
            if case_sensitive:
                return match_text not in input_text
            return not any(str(item).lower() == match_text.lower() for item in input_text)

        if operator in ["less than", "less than or equal", "greater than", "greater than or equal"]:
            # Compare list length with match_text as number
            try:
                list_length = len(input_text)
                match_num = float(match_text)
                if operator == "less than":
                    return list_length < match_num
                if operator == "less than or equal":
                    return list_length <= match_num
                if operator == "greater than":
                    return list_length > match_num
                if operator == "greater than or equal":
                    return list_length >= match_num
            except ValueError:
                return False  # Invalid number format for comparison

        if operator == "equals":
            # Check if list length equals match_text as number
            try:
                return len(input_text) == float(match_text)
            except ValueError:
                return False

        if operator == "not equals":
            # Check if list length does not equal match_text as number
            try:
                return len(input_text) != float(match_text)
            except ValueError:
                return False

    # Handle Dict operations
    if isinstance(input_text, dict):
        if operator == "contains":
            # Check if match_text is a key in the dict
            if case_sensitive:
                return match_text in input_text
            return any(str(key).lower() == match_text.lower() for key in input_text)

        if operator == "not contains":
            # Check if match_text is NOT a key in the dict
            if case_sensitive:
                return match_text not in input_text
            return not any(str(key).lower() == match_text.lower() for key in input_text)

        if operator in ["less than", "less than or equal", "greater than", "greater than or equal"]:
            # Compare dict length (number of keys) with match_text as number
            try:
                dict_length = len(input_text)
                match_num = float(match_text)
                if operator == "less than":
                    return dict_length < match_num
                if operator == "less than or equal":
                    return dict_length <= match_num
                if operator == "greater than":
                    return dict_length > match_num
                if operator == "greater than or equal":
                    return dict_length >= match_num
            except ValueError:
                return False  # Invalid number format for comparison

        if operator == "equals":
            # Check if dict length equals match_text as number
            try:
                return len(input_text) == float(match_text)
            except ValueError:
                return False

        if operator == "not equals":
            # Check if dict length does not equal match_text as number
            try:
                return len(input_text) != float(match_text)
            except ValueError:
                return False

    # Handle String operations (existing logic)
    if isinstance(input_text, str):
        # Convert to string if it's not already
        input_text = str(input_text)

        if not case_sensitive and operator != "regex":
            input_text = input_text.lower()
            match_text = match_text.lower()

        if operator == "equals":
            return input_text == match_text
        if operator == "not equals":
            return input_text != match_text
        if operator in ["contains", "in"]:
            return match_text in input_text
        if operator in ["not contains", "not in"]:
            return match_text not in input_text
        if operator == "starts with":
            return input_text.startswith(match_text)
        if operator == "ends with":
            return input_text.endswith(match_text)
        if operator == "regex":
            try:
                return bool(re.match(match_text, input_text))
            except re.error:
                return False  # Return False if the regex is invalid
        if operator in ["less than", "less than or equal", "greater than", "greater than or equal"]:
            try:
                input_num = float(input_text)
                match_num = float(match_text)
                if operator == "less than":
                    return input_num < match_num
                if operator == "less than or equal":
                    return input_num <= match_num
                if operator == "greater than":
                    return input_num > match_num
                if operator == "greater than or equal":
                    return input_num >= match_num
            except ValueError:
                return False  # Invalid number format for comparison

    return False



def detect_and_register_cond_edges(
    builder,
    cond_edge_name,
    previous_node,
    conditional_func: Callable,
    send_node_name: str | None = None,
):
    """Detect previous_nodes connections and add them to conditional edge."""
    if not builder:
        msg = f"{cond_edge_name} | No StateGraph builder found in context."
        raise ValueError(msg)

    if previous_node:
        # Create conditional entry point if previous node is from CreateStateGraphComponent
        if send_node_name:
            if previous_node.__class__.__name__ == "CreateStateGraphComponent":
                builder.add_conditional_edges(START, conditional_func, [send_node_name])
                print(f"Added send api edge: START -> {conditional_func.__name__}")  # noqa: T201
            else:
                # Get the node name from the previous GraphNode component
                prev_node_name = previous_node.node_name
                builder.add_conditional_edges(prev_node_name, conditional_func, [send_node_name])
                print(f"Added send api edge: {prev_node_name} -> {conditional_func.__name__}")  # noqa: T201
        # Create conditional entry point if previous node is from CreateStateGraphComponent
        elif previous_node.__class__.__name__ == "CreateStateGraphComponent":
            builder.add_conditional_edges(START, conditional_func)
            print(f"Added conditional edge: START -> {conditional_func.__name__}")  # noqa: T201
        else:
            # Get the node name from the previous GraphNode component
            prev_node_name = previous_node.node_name
            builder.add_conditional_edges(prev_node_name, conditional_func)
            print(f"Added conditional edge: {prev_node_name} -> {conditional_func.__name__}")  # noqa: T201
    else:
        msg = f"{cond_edge_name} | No previous node found, please connect one..."
        raise ValueError(msg)
