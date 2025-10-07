from collections.abc import Callable  # noqa: N999
from typing import Any

from langgraph.graph import START
from langgraph.types import Send
from pydantic import Field, create_model

from langflow.custom.custom_component.component import Component
from langflow.io import DictInput, HandleInput, MessageTextInput, Output


class SendMapReduceForLangGraph(Component):
    display_name = "Send API For LangGraph"
    description = "Use this component to send API in LangGraph."
    documentation: str = "https://langchain-ai.github.io/langgraph/how-tos/graph-api/#map-reduce-and-the-send-api"
    icon = "LangChain"
    name = "SendMapReduceForLangGraph"

    inputs = [
        HandleInput(
            name="previous_node",
            display_name="Previous Node",
            info="Connect previous GraphNode to create conditional edge. "
            "If State Graph is connected instead of GraphNode, this will be treated as the Conditional Entry Point. "
            "If using with Command Addon, this field will not be used because it will depend on the "
            "GraphNode the Command Addon linked to.",
            input_types=[
                "GraphNodeAsSubGraph",
                "GraphNodeForCrewAIAgent",
                "GraphNodeForAgent",
                "GraphNodeForCrewAICrew",
                "GraphNodeForFunction",
                "CreateStateGraphComponent"],
        ),
        HandleInput(
            name="input_state",
            display_name="Input State",
            info="Input State Of Send API. "
            "If using with Command Addon, this field will not be used because it will depend on the GraphNode "
            "the Command Addon linked to.",
            input_types=["ModelClassWrapper"],
        ),
        MessageTextInput(
            name="send_api_name",
            display_name="Name Of The Send API",
            info="Name of the send API",
            required=True,
        ),
        MessageTextInput(
            name="send_node_name",
            display_name="Send Node Name",
            info="Specify the name of the node to send to",
            required=True,
        ),
        MessageTextInput(
            name="state_field_list_objects",
            display_name="Input State Field (List of Objects)",
            info="Enter the name of the field in the input state that holds a list of objects",
            required=True,
        ),
        MessageTextInput(
            name="object_name",
            display_name="Object Field",
            info="Specify the field name of each object. "
            "This becomes the first part of the value sent to the target node. "
            "Each object in the list will be sent "
            "one by one as the new input state, overriding the node's current state.",
            required=True,
        ),
        HandleInput(
            name="custom_structure_data",
            display_name="Custom Class",
            info="Connect a custom classes (via structure data component) to use their instance when create input data",
            input_types=["ModelClassWrapper"],
            is_list=True,
            advanced=True,
        ),
        DictInput(
            name="extra_input",
            display_name="Extra Input Fields",
            info="Provide additional key-value pairs to send along with each object. This forms the second part of the "
            "value sent to the target node. The same extra input is included for every object in the list.\n"
            "You can specify the extra input for the Send API based on these syntax below.\n\n"
            "• Use @is_field field_name to access the value of the field in input state.\n"
            "Example: @is_field field_name\n\n"
            "• Use @is_list to mark a list.\n"
            "Example: @is_list [1, 2, 3]\n\n"
            "• Use @is_dict to mark a dictionary.\n"
            'Example: @is_dict {"name": "John", "age": 30}\n\n'
            "• Use @is_class to mark a custom class type.\n"
            'Example: @is_class Company{"name": "TechCorp", "employees": 100, "public": false}\n\n'
            "• Use @is_bool to mark a boolean value.\n"
            "Example: @is_bool True\n\n"
            "• For nested structures (like a list or dict of custom classes or involve in using node output), "
            "define each class instance or retrieve value from node output first, assign it to a variable using "
            "as {variable}, and then reference those variables inside your list, dict, or another class. "
            "Separate multiple declarations using ||.\n"
            'Example: @is_field person as p1 || @is_class Person{"name": "a", "age": 8} as p2 || @is_class '
            'Person{"name": "b", "age": 9} as p3 || @is_list [p1, p2, p3]\n'
            '@is_class Birthday{"date": 2, "month": 12, "year": 1994} as birthday || @is_class Person{"name": "Lan", '
            '"age": 20, "birthday": birthday} as person || @is_list [person]\n\n'
            "• For simple types like strings, integers, and floats, just input the value directly.\n"
            'Examples: "Hello", 1, 1.0\n\n'
            "Note: Use double quotes for strings and dict keys.\n"
            "If use this  with Command Addon and the pair in extra input share the same key with the pair from update "
            "value, the value from extra input will overwrite the value in update value.",
        )
    ]

    outputs = [
        Output(
            display_name="Send AI For Node",
            name="send_to_node",
            method="build_api",
            info="This is just for visualization purposes, it does not actually build a Send API. "
            "But you will need to connect this to target node to add that node to the graph and complete the flow.",

        ),
        Output(
            display_name="Send API For Command",
            name="send_for_command",
            method="create_send_for_command",
            info="Connect this to Command Addon to be able to use this Send API in Command.",
        ),
    ]


    def _pre_run_setup(self):
        if self.previous_node:
            self.graph_input_state = self.previous_node.graph_input_state
            self.graph_builder = self.previous_node.graph_builder
            self.graph_output_state = self.previous_node.graph_output_state
            self.graph_context_schema = self.previous_nodes.graph_context_schema

            # If input_state is not provided, get from shared data
            if not self.input_state:
                self.input_state = self.graph_input_state
            self.input_model = self.input_state.model_class
            print(f"Input model for SendMapReduceForLangGraph: {self.input_model}")  # noqa: T201

        """Pre-run setup to normalize input data and prepare custom models"""
        self.custom_models = {}
        custom_model_names = []
        # Get available custom models from custom_structure_data
        if self.custom_structure_data:
            for model_wrapper in self.custom_structure_data:
                if hasattr(model_wrapper, "model_class") and hasattr(model_wrapper, "schema"):
                    class_name = model_wrapper.schema.get("class_name", "")
                    if class_name:
                        self.custom_models[class_name] = model_wrapper.model_class
                        custom_model_names.append(class_name)

        # Print available custom types to help the user
        if custom_model_names:
            print(f"Available custom types: {', '.join(custom_model_names)}")  # noqa: T201


    def _detect_and_register_edges(self, conditional_func: Callable):
        """Detect previous_nodes connections and add them to conditional edge."""
        builder = self.graph_builder
        if not builder:
            msg = f"{self.send_api_name} | No StateGraph builder found in context."
            raise ValueError(msg)

        if self.previous_node:
            # Create conditional entry point if previous node is from CreateStateGraphComponent
            if self.previous_node.__class__.__name__ == "CreateStateGraphComponent":
                builder.add_conditional_edges(START, conditional_func, [self.send_node_name])
                print(f"Added send api edge: START -> {conditional_func.__name__}")  # noqa: T201
            else:
                # Get the node name from the previous GraphNode component
                prev_node_name = self.previous_node.node_name
                builder.add_conditional_edges(prev_node_name, conditional_func, [self.send_node_name])
                print(f"Added send api edge: {prev_node_name} -> {conditional_func.__name__}")  # noqa: T201
        else:
            msg = f"{self.send_api_name} | No previous node found, please connect one..."
            raise ValueError(msg)

    def _determine_object_type(self, object_list):
        """Determine the type of objects in the list."""
        if not object_list:
            return str  # Fallback to str if empty

        first_item = object_list[0]

        # Check for basic Python types
        if isinstance(first_item, str):
            return str
        if isinstance(first_item, int):
            return int
        if isinstance(first_item, float):
            return float
        if isinstance(first_item, bool):
            return bool
        if isinstance(first_item, dict):
            return dict
        if isinstance(first_item, list):
            return list

        # Check for Pydantic models
        if (
            hasattr(first_item, "model_dump") or
            hasattr(first_item, "__dataclass_fields__") or
            hasattr(first_item, "__class__")
            ):
            return first_item.__class__

        # Ultimate fallback
        return Any


    def create_send_for_command(self) -> dict:
        """Create a Send API for Command Addon."""
        # Set type_hint
        type_hint = self.input_model if self.previous_node else Any

        def send_api_edge(state: type_hint, update_value_from_command: dict | None = None):  # type: ignore  # noqa: PGH003
            if update_value_from_command:
                if self.extra_input:
                    # Normalize extra input data
                    self.extra_input = self.normalize_input_data(state, self.extra_input)
                    self.extra_input = {**update_value_from_command, **self.extra_input}
                else:
                    self.extra_input = update_value_from_command
            elif self.extra_input:
                self.extra_input = self.normalize_input_data(state, self.extra_input)

            # Get the list of objects to inspect their type
            object_list = getattr(state, self.state_field_list_objects)

            # Determine the object type dynamically
            object_type = self._determine_object_type(object_list)

            # Determine the type of value in extra_input
            if self.extra_input:
                extra_input_fields = {}
                for k, v in self.extra_input.items():
                    # Get the type of the value, not the value itself
                    field_type = type(v)
                    extra_input_fields[k] = (
                        field_type,
                        Field(description=f"Extra field {k} of type {field_type.__name__} to be sent to the node.")
                    )

                # Create a sending object with dynamic type
                sending_object = create_model(
                    "SendingObject",
                    **{
                        self.object_name: (
                            object_type,
                            Field(description=f"Object of type {object_type.__name__} to be sent to the node.")
                        )
                    },
                    **extra_input_fields
                )
                result = [
                    Send(
                        self.send_node_name,
                        sending_object(**{self.object_name: s, **dict(self.extra_input.items())})
                    ) for s in object_list
                ]

            else:
                sending_object = create_model(
                    "SendingObject",
                    **{self.object_name: (
                        object_type,
                        Field(description=f"Object of type {object_type.__name__} to be sent to the node.")
                    )}
                )
                result = [Send(self.send_node_name, sending_object(**{self.object_name: s})) for s in object_list]
            print("Debug send api result:", result)  # noqa: T201
            return result

        print(f"Building send api: {self.send_api_name}")  # noqa: T201
        send_api_edge.__name__ = self.send_api_name
        send_api_edge.__qualname__   = self.send_api_name

        return {self.send_api_name: send_api_edge}

    def build_api(self) -> "SendMapReduceForLangGraph":
        send_api_edge = self.create_send_for_command().get(self.send_api_name)
        # Register the conditional edge with the builder
        self._detect_and_register_edges(send_api_edge)
        return self






    ############## Input Normalization and Custom Model Handling ##############
    def normalize_input_data(self, dict_or_object=None, input_data=None):
        """Normalize input data based on type flags.

        dict_or_object:
            This will be used to support @is_field. Get the field in dict_or_object.
            Could be things like node output, state, etc.

        input_data:
            A dictionary which key is normal string and value
            is a declaration string that need to be converted to the correct type.
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
                    input_data[key] = self._process_single_declaration(dict_or_object, value_)

                elif len(declarations) > 1:
                    input_data[key] = self._process_nested_declaration(dict_or_object, declarations)

                print(f"✓ Converted {key}: {original_value} -> {input_data[key]} ({type(input_data[key]).__name__})")  # noqa: T201

            except Exception as e:
                msg = f"✗ Error converting {key}: {e} || Check your prompt, connection again."
                raise ValueError(msg) from e
        return input_data


    def _parse_class_instance(self, value: str, variables: dict):
        """Parse class instance from string like 'Person{"name": "John", "age": 30}'."""
        # Extract class name and parameters
        if "{" not in value or "}" not in value:
            msg = f'Invalid class format: {value}. Expected format: ClassName{{"key":"value", "key2":value2}}'
            raise ValueError(msg)

        class_name = value.split("{")[0].strip()
        params_str = value.split("{")[1].split("}")[0].strip()

        # Only replace variables if there are variables to replace
        if variables:
            params_str = self._replace_variables_in_json(f"{{{params_str}}}", variables)
            print(f"After replacing variables: {params_str}")  # noqa: T201
        else:
            params_str = f"{{{params_str}}}"

        # Check if class exists in custom_models
        if class_name not in self.custom_models:
            available_classes = list(self.custom_models.keys())
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
        model_class = self.custom_models[class_name]
        return model_class(**params)

    def _parse_field(self, dict_or_object, value: str):
        return getattr(dict_or_object, value) or dict_or_object.get(value)

    def _parse_bool(self, value: str) -> bool:
        """Parse boolean from string."""
        value = value.lower().strip()
        if value in ["true", "1", "yes", "on"]:
            return True
        if value in ["false", "0", "no", "off"]:
            return False
        msg = f"Cannot convert '{value}' to boolean"
        raise ValueError(msg)

    def _safe_literal_eval(self, value: str, expected_type=None):
        """Safely evaluate literal with type checking."""
        import ast
        result = ast.literal_eval(value)
        if expected_type and not isinstance(result, expected_type):
            msg = f"Expected {expected_type.__name__}, got {type(result).__name__}"
            raise ValueError(msg)
        return result

    def _process_single_declaration(self, dict_or_object, value: str):
        """Process a single declaration."""
        if value.startswith("@is_list"):
            clean_value = value.replace("@is_list", "").strip()
            return self._safe_literal_eval(clean_value, expected_type=list)

        if value.startswith("@is_dict"):
            clean_value = value.replace("@is_dict", "").strip()
            return self._safe_literal_eval(clean_value, expected_type=dict)

        if value.startswith("@is_class"):
            clean_value = value.replace("@is_class", "").strip()
            # Remove "as variable_name" if present for single declarations
            if " as " in clean_value:
                clean_value = clean_value.split(" as ")[0].strip()
            return self._parse_class_instance(clean_value, {})

        if value.startswith("@is_bool"):
            clean_value = value.replace("@is_bool", "").strip()
            return self._parse_bool(clean_value)

        if value.startswith("@is_field"):
            if not dict_or_object:
                msg = "No dictionary or object provided for @is_field. Check your code."
                raise ValueError(msg)
            clean_value = value.replace("@is_field", "").strip()
            return self._parse_field(dict_or_object, clean_value)

        return value


    def _process_nested_declaration(self, dict_or_object, declarations: list):
        """Process nested declarations with variable tracking."""
        variables = {}  # Store variables for this field
        # Multiple nested declarations - process in order and track variables
        for d in declarations:
            d_ = d.strip()

            if d_.startswith("@is_class"):
                clean_value = d_.replace("@is_class", "").strip()
                # Check if there's an "as variable_name" part
                if " as " in clean_value:
                    class_part, var_name = clean_value.split(" as ", 1)
                    var_name = var_name.strip()
                    # Remove "as var_name" from class_part for parsing
                    instance = self._parse_class_instance(class_part.strip(), variables)
                    variables[var_name] = instance
                    print(f"Created variable '{var_name}': {instance}")  # noqa: T201
                else:
                    msg = "Class declaration must include 'as variable_name' part."
                    raise ValueError(msg)

            elif d_.startswith("@is_field"):
                if not dict_or_object:
                    msg = "No dictionary or object provided for @is_field. Check your code."
                    raise ValueError(msg)
                clean_value = d_.replace("@is_field", "").strip()
                if " as " in clean_value:
                    node_output_value_field, var_name = clean_value.split(" as ", 1)
                    var_name = var_name.strip()
                    node_output_value = self._parse_field(dict_or_object, node_output_value_field.strip())
                    variables[var_name] = node_output_value
                    print(f"Created variable '{var_name}': {node_output_value}")  # noqa: T201
                else:
                    msg = "Node output declaration must include 'as variable_name' part."
                    raise ValueError(msg)

            elif d_.startswith("@is_list"):
                clean_value = d_.replace("@is_list", "").strip()
                # Replace variable references in the list
                processed_value = self._replace_variables_in_json(clean_value, variables)

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
                    result = self._safe_literal_eval(processed_value, expected_type=list)

            elif d_.startswith("@is_dict"):
                clean_value = d_.replace("@is_dict", "").strip()
                # Replace variable references in the dict
                processed_value = self._replace_variables_in_json(clean_value, variables)

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
                    result = self._safe_literal_eval(processed_value, expected_type=dict)
        return result


    def _replace_variables_in_json(self, json_str: str, variables: dict) -> str:
        """Replace variable references in JSON string with actual values."""
        import json
        import re

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
