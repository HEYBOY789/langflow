import json  # noqa: INP001
import re

from src.backend.base.langflow.components.LangGraph.utils.agent_result_func import safe_literal_eval


def normalize_input_data(input_data=None, dict_or_object=None, parent_state=None):
    """Normalize input data based on type flags.

    input_data: A dictionary which key is normal string and value is a declaration
    string that need to be converted to the correct type.

    dict_or_object: This will be used to support @is_field. Get the field in dict_or_object.
    Could be things like node output, state, etc.
    """
    print("Starting input data normalization...")  # noqa: T201

    if input_data is None:
        return {}

    for key, value in input_data.items():
        original_value = value

        try:
            # Split declaration. If there are multiple declarations, they will be divided by ||
            # Ex: @is_class Person{"name"="a", "age"=8} as p || @is_list [p]
            declarations = value.split("||")

            if len(declarations) == 1:
                # Single declaration - process normally
                value_ = declarations[0].strip()
                input_data[key] = process_single_declaration(value_, dict_or_object, parent_state)

            elif len(declarations) > 1:
                input_data[key] = process_nested_declaration(declarations, dict_or_object, parent_state)

            print(f"✓ Converted {key}: {original_value} -> {input_data[key]} ({type(input_data[key]).__name__})")  # noqa: T201

        except (ValueError, KeyError, AttributeError, TypeError) as e:
            print(f"✗ Error converting {key}: {e} || Check your prompt again. Keeping original value.")  # noqa: T201
            input_data[key] = original_value
    return input_data


def parse_class_instance(custom_models, value: str, variables: dict):
    """Parse class instance from string like 'Person{"name": "John", "age": 30}'."""
    # Extract class name and parameters
    if "{" not in value or "}" not in value:
        msg = f'Invalid class format: {value}. Expected format: ClassName{{"key":"value", "key2":value2}}'
        raise ValueError(msg)

    class_name = value.split("{")[0].strip()
    params_str = value.split("{")[1].split("}")[0].strip()

    # Only replace variables if there are variables to replace
    if variables:
        params_str = replace_variables_in_json(f"{{{params_str}}}", variables)
        print(f"After replacing variables: {params_str}")  # noqa: T201
    else:
        params_str = f"{{{params_str}}}"

    # Check if class exists in custom_models
    if class_name not in custom_models:
        available_classes = list(custom_models.keys())
        msg = f"Class '{class_name}' not found. Available classes: {available_classes}"
        raise ValueError(msg)

    # Parse parameters as JSON
    params = {}
    if params_str:
        import json
        try:
            params = json.loads(params_str)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON format in class parameters: {params_str}. Error: {e}"
            raise ValueError(msg) from e

    # Create instance
    model_class = custom_models[class_name]
    return model_class(**params)


def parse_field(dict_or_object, value: str):
    return getattr(dict_or_object, value) or dict_or_object.get(value)

def parse_parent_state(parent_state, value: str):
    if not parent_state:
        msg = "Parent state is not set. Make sure to pass it from GraphNodeAsSubGraph."
        raise ValueError(msg)
    return parse_field(parent_state, value)

def parse_bool(value: str) -> bool:
    """Parse boolean from string."""
    value = value.lower().strip()
    if value in ["true", "1", "yes", "on"]:
        return True
    if value in ["false", "0", "no", "off"]:
        return False
    msg = f"Cannot convert '{value}' to boolean"
    raise ValueError(msg)

def process_single_declaration(value: str, dict_or_object=None, parent_state=None):
    """Process a single declaration."""
    if value.startswith("@is_list"):
        clean_value = value.replace("@is_list", "").strip()
        return safe_literal_eval(clean_value, expected_type=list)

    if value.startswith("@is_dict"):
        clean_value = value.replace("@is_dict", "").strip()
        return safe_literal_eval(clean_value, expected_type=dict)

    if value.startswith("@is_class"):
        clean_value = value.replace("@is_class", "").strip()
        # Remove "as variable_name" if present for single declarations
        if " as " in clean_value:
            clean_value = clean_value.split(" as ")[0].strip()
        return parse_class_instance(clean_value, {})

    if value.startswith("@is_bool"):
        clean_value = value.replace("@is_bool", "").strip()
        return parse_bool(clean_value)

    if dict_or_object and value.startswith("@is_field"):
        clean_value = value.replace("@is_field", "").strip()
        return parse_field(dict_or_object, clean_value)

    if parent_state and value.startswith("@is_parent_state"):
        clean_value = value.replace("@is_parent_state", "").strip()
        return parse_parent_state(parent_state, clean_value)

    return value


def process_nested_declaration(declarations: list, dict_or_object=None, parent_state=None):
    """Process nested declarations with variable tracking."""
    variables = {}  # Store variables for this field
    # Multiple nested declarations - process in order and track variables
    for declare in declarations:
        d = declare.strip()

        if d.startswith("@is_class"):
            clean_value = d.replace("@is_class", "").strip()

            # Check if there's an "as variable_name" part
            if " as " in clean_value:
                class_part, var_name = clean_value.split(" as ", 1)
                var_name = var_name.strip()
                # Remove "as var_name" from class_part for parsing
                instance = parse_class_instance(class_part.strip(), variables)
                variables[var_name] = instance
                print(f"Created variable '{var_name}': {instance}")  # noqa: T201
            else:
                msg = "Class declaration must include 'as variable_name' part."
                raise ValueError(msg)

        if dict_or_object and d.startswith("@is_field"):
            clean_value = d.replace("@is_field", "").strip()
            if " as " in clean_value:
                node_output_value_field, var_name = clean_value.split(" as ", 1)
                var_name = var_name.strip()
                node_output_value = parse_field(dict_or_object, node_output_value_field.strip())
                variables[var_name] = node_output_value
                print(f"Created variable '{var_name}': {node_output_value}")  # noqa: T201
            else:
                msg = "Node output declaration must include 'as variable_name' part."
                raise ValueError(msg)

        elif parent_state and d.startswith("@is_parent_state"):
            clean_value = d.replace("@is_parent_state", "").strip()
            if " as " in clean_value:
                parent_field, var_name = clean_value.split(" as ", 1)
                var_name = var_name.strip()
                parent_value = parse_parent_state(parent_state, parent_field.strip())
                variables[var_name] = parent_value
                print(f"Created variable '{var_name}': {parent_value}")  # noqa: T201
            else:
                msg = "Parent state declaration must include 'as variable_name' part."
                raise ValueError(msg)

        elif d.startswith("@is_list"):
            clean_value = d.replace("@is_list", "").strip()
            # Replace variable references in the list
            processed_value = replace_variables_in_json(clean_value, variables)

            # Use json.loads if variables were replaced, otherwise use ast.literal_eval
            if variables:
                import json
                try:
                    result = json.loads(processed_value)
                    if not isinstance(result, list):
                        msg = f"Expected list, got {type(result).__name__}"
                        raise ValueError(msg)  # noqa: TRY004
                except json.JSONDecodeError as e:
                    msg = f"Invalid JSON in list: {processed_value}. Error: {e}"
                    raise ValueError(msg) from e
            else:
                result = safe_literal_eval(processed_value, expected_type=list)

        elif d.startswith("@is_dict"):
            clean_value = d.replace("@is_dict", "").strip()
            # Replace variable references in the dict
            processed_value = replace_variables_in_json(clean_value, variables)

            # Use json.loads if variables were replaced, otherwise use ast.literal_eval
            if variables:
                import json
                try:
                    result = json.loads(processed_value)
                    if not isinstance(result, dict):
                        msg = f"Expected dict, got {type(result).__name__}"
                        raise ValueError(msg)  # noqa: TRY004
                except json.JSONDecodeError as e:
                    msg = f"Invalid JSON in dict: {processed_value}. Error: {e}"
                    raise ValueError(msg) from e
            else:
                result = safe_literal_eval(processed_value, expected_type=dict)
    return result


def replace_variables_in_json(json_str: str, variables: dict) -> str:
    """Replace variable references in JSON string with actual values."""

    def replace_if_variable(match):
        word = match.group(1)

        # Check if this word is a variable we know about
        if word in variables:
            # Convert the variable value to its JSON representation
            var_value = variables[word]
            if hasattr(var_value, "model_dump"):
                # Pydantic model - convert to dict then to JSON
                return json.dumps(var_value.model_dump())
            # Other types - convert directly to JSON
            return json.dumps(var_value)
        # Not a variable, keep as is
        return word

    # Split by strings to avoid replacing variables inside quoted strings
    parts = re.split(r'("[^"]*"|\'[^\']*\')', json_str)
    result_parts = []
    for i, part in enumerate(parts):
        if i % 2 == 0:  # Outside of strings
            # Apply variable replacement only to parts outside of quoted strings
            # Pattern to match unquoted words that could be variables
            pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
            part_ = re.sub(pattern, replace_if_variable, part)
            result_parts.append(part_)
        else:  # Inside strings (i % 2 == 1), keep as is
            result_parts.append(part)

    return "".join(result_parts)

