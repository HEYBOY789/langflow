import json  # noqa: N999
import re
from collections.abc import Callable, Sequence

from langgraph.graph import END
from langgraph.types import Command, Send

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, DictInput, DropdownInput, HandleInput, MessageTextInput, NestedDictInput, Output


class CommandAddonForLangGraph(Component):
    display_name = "Command Addon For Graph Node"
    description = ("Use this component with GraphNode to create a GraphNode that returning a Command object."
    "Result of the node (agent output, function output) will be used to evaluate to return a Command object.")
    documentation: str = "https://langchain-ai.github.io/langgraph/how-tos/graph-api/#combine-control-flow-and-state-updates-with-command"
    icon = "LangChain"
    name = "CommandAddonForLangGraph"

    inputs = [
        DropdownInput(
            name="condition_type",
            display_name="Condition Type",
            info=("Choose either built-in or custom condition type. "
            "If you only need to check 1 value and there is a suitable operator, use built-in condition type. "
            "If you need to check multiple values or use custom logic, use custom condition type."),
            options=["Built-in", "Custom"],
            required=True,
            value="Built-in",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="state_field",
            display_name="Field Name Of The Result",
            info="Specify the field name in the result that you want to use for evaluation.",
            required=True,
            dynamic=True,
            show=True
        ),
        DropdownInput(
            name="operator",
            display_name="Operator",
            info=("Choose an operator to compare the input state value with the condition value. "
            "Beside string and number, you can also check list and dict. "
            "For a list, you can check its length or if it contains a specific value. "
            "For a dict, you can check its length or if it contains a specific key."),
            options=[
                "equals", "not equals", "less than", "less than or equal", "greater than", "greater than or equal",
                "contains", "not contains", "starts with", "ends with","regex"
            ],
            required=True,
            show=True,
            value="equals",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="match_value",
            display_name="Match Value",
            info="Specify the value to compare with the input state value.",
            required=True,
            dynamic=True,
            show=True
        ),
        HandleInput(
            name="custom_structure_data",
            display_name="Custom Class",
            info="Connect a custom classes (via structure data component) to use their instance when create input data",
            input_types=["ModelClassWrapper"],
            is_list=True,
            advanced=True,
        ),
        BoolInput(
            name="case_sensitive",
            display_name="Case Sensitive",
            info="If true, the comparison will be case sensitive.",
            value=True,
            advanced=True,
        ),
        DropdownInput(
            name="goto_type_true",
            display_name="True Route Goto Type",
            info="Choose either normal GraphNode or Send API to route to when condition is true",
            real_time_refresh=True,
            options=["GraphNode", "Send API"],
            value="GraphNode"
        ),
        HandleInput(
            name="true_route_send",
            display_name="Connect Send API For True Route",
            info="Connect a Send API component to route to when the condition is true. ",
            dynamic=True,
            show=False,
            input_types=["Dict"],
            required=False,
        ),
        MessageTextInput(
            name="true_route_go_to",
            display_name="True Route Go To",
            info=("Specify the goto node when condition is true."
            "To route to multiple nodes, use commas to separate them."
            "Use 'END' to route to the end of the graph."),
            required=True,
            dynamic=True,
            show=True,
            value="goto_node_true_1, goto_node_true_2"
        ),
        DictInput(
            name="true_route_update_value",
            display_name="True Route Update Value",
            info="Specify the update value when the condition is true.\n"
            "Specify the update_value (must be a dictionary written as a string). "
            "The update_value will replace the current input state of the target node. "
            "When used with Send API, the value sent by the API will override the update_value.\n"
            "You can specify the update data for the command based on these syntax below.\n\n"
            "• Use @is_field field_name to access the value of the field in node output.\n"
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
            "define each class instance or retrieve value from node output first, "
            "assign it to a variable using as {variable}, and then reference those variables inside your list, dict, "
            "or another class. Separate multiple declarations using ||.\n"
            'Example: @is_field person as p1 || @is_class Person{"name": "a", "age": 8} as p2 || '
            '"@is_class Person{"name": "b", "age": 9} as p3 || @is_list [p1, p2, p3]\n'
            '@is_class Birthday{"date": 2, "month": 12, "year": 1994} as birthday'
            ' || @is_class Person{"name": "Lan", "age": 20, "birthday": birthday} as person || @is_list [person]\n\n'
            "• For simple types like strings, integers, and floats, just input the value directly\n."
            'Examples: "Hello", 1, 1.0\n\n'
            "Note: Use double quotes for strings and dict keys.\n"
            "If use this Send API, the update value will also be sent to the target node as extra input.",
            required=True,
            dynamic=True,
            show=True,
            is_list=True,
            value={"true_update_val_key_1": "true_update_val_val_1", "true_update_val_key_2": "{node_output}"}
        ),
        DropdownInput(
            name="goto_type_false",
            display_name="False Route Goto Type",
            info="Choose either normal GraphNode or Send API to route to when condition is false",
            real_time_refresh=True,
            options=["GraphNode", "Send API"],
            value="GraphNode"
        ),
        HandleInput(
            name="false_route_send",
            display_name="Connect Send API For False Route",
            info="Connect a Send API component to route to when the condition is false. ",
            dynamic=True,
            show=False,
            input_types=["Dict"],
            required=False,
        ),
        MessageTextInput(
            name="false_route_go_to",
            display_name="False Route Go To",
            info="Specify the goto node when condition is false. "
            "To route to multiple nodes, use commas to separate them. "
            "Use 'END' to route to the end of the graph.",
            required=True,
            dynamic=True,
            show=True,
            value="goto_node_false_1, goto_node_false_2"
        ),
        DictInput(
            name="false_route_update_value",
            display_name="False Route Update Value",
            info="Specify the update value when the condition is false.\n"
            "Specify the update_value (must be a dictionary written as a string). "
            "The update_value will replace the current input state of the target node. "
            "When used with Send API, the value sent by the API will override the update_value.\n"
            "You can specify the update data for the command based on these syntax below.\n\n"
            "• Use @is_field field_name to access the value of the field in node output.\n"
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
            "define each class instance or retrieve value from node output first,"
            " assign it to a variable using as {variable}, "
            "and then reference those variables inside your list, dict, or another class. "
            "Separate multiple declarations using ||.\n"
            'Example: @is_field person as p1 || @is_class Person{"name": "a", "age": 8} as p2 || '
            '@is_class Person{"name": "b", "age": 9} as p3 || @is_list [p1, p2, p3]\n'
            '@is_class Birthday{"date": 2, "month": 12, "year": 1994} as birthday'
            ' || @is_class Person{"name": "Lan", "age": 20, "birthday": birthday} as person || @is_list [person]\n\n'
            "• For simple types like strings, integers, and floats, just input the value directly\n."
            'Examples: "Hello", 1, 1.0\n\n'
            "Note: Use double quotes for strings and dict keys.\n"
            "If use this Send API, the update value will also be sent to the target node as extra input.",
            required=True,
            dynamic=True,
            show=True,
            is_list=True,
            value={"false_update_val_key_1": "false_update_val_val_1", "false_update_val_key_2": "{node_output}"}
        ),
        HandleInput(
            name="custom_condition",
            input_types=["Callable"],
            display_name="Function",
            info=(
                "Connect a Python function to use its custom logic. Must return var_name. Must have one input "
                "reference to the output of the node (agent output, function output). Use node_output.field to access "
                "the fields in node_output. Make sure the fields exist in the output model of the node.\n"
                "Example:\n"
                "def conditional_edge(node_output):\n"
                "____if node_output.field == 'condi_1':\n"
                "________return 'var_name_1'\n"
                "____elif node_output.field == 'condi_2':\n"
                "________return 'var_name_2'\n"
                "____else:\n"
                "________return 'var_name_3'\n"
            ),
            required=False,
            show=False,
            dynamic=True,
        ),
        HandleInput(
            name="send_api_custom",
            display_name="Connect Send API To Route To",
            info="Connect a Send API components to route to when the condition is custom. "
            "Use {send_api_name} to refer to the name of the send API in the Goto section",
            dynamic=True,
            show=False,
            input_types=["Dict"],
            is_list=True
        ),
        DictInput(
            name="goto_custom",
            display_name="Goto",
            info="Specify var_name with either node_name or {send_api_name};"
            " Separate multiple node_name with commas (e.g., node_1, node_2, end),"
            " and use end to refer to END (multiple is not valid for send_api_name).",
            required=False,
            dynamic=True,
            show=False,
            value={"var_name_1": "node_name_1, node_name_2", "var_name_2": "{send_node_name_2}", "var_name_3": "node_name_3"},  # noqa: E501
            is_list=True,
        ),
        NestedDictInput(
            name="update_value",
            display_name="Update Value",
            info="Specify both var_name and update_value (where update_value must be a dictionary written as a string)."
            " The update_value will replace the current input state of the target node. "
            "When used with Send API, the value sent by the API will override the update_value.\n"
            "You can specify the update data for the command based on these syntax below.\n\n"
            "• Use @is_field field_name to access the value of the field in node output.\n"
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
            "define each class instance or retrieve value from node output first,"
            " assign it to a variable using as {variable}, "
            "and then reference those variables inside your list, dict, or another class. "
            "Separate multiple declarations using ||.\n"
            'Example: @is_field person as p1 || @is_class Person{"name": "a", "age": 8} as p2 || '
            '@is_class Person{"name": "b", "age": 9} as p3 || @is_list [p1, p2, p3]\n'
            '@is_class Birthday{"date": 2, "month": 12, "year": 1994} as birthday'
            ' || @is_class Person{"name": "Lan", "age": 20, "birthday": birthday} as person || @is_list [person]\n\n'
            "• For simple types like strings, integers, and floats, just input the value directly.\n"
            'Examples: "Hello", 1, 1.0\n\n'
            "Note: Use double quotes for strings and dict keys.\n"
            "Your custom code must return var_name, which will be associated with the update_value.\n"
            "If use this  with Send API, the update value will also be sent to the target node as extra input. "
            "If update value have a pair that share the same key with extra input,"
            " the value from extra input will overwrite the value in update value.",
            required=False,
            dynamic=True,
            show=False,
            value={
                "var_name_1": {"update_val_key_1": "update_val_val_1"}, "var_name_2": {"update_val_key_2": "@is_field field_name_2"},  # noqa: E501
                "var_name_3": {"update_val_key_3": "@is_list [1, 2, 3]"}
            },
            is_list=True,
        ),
        BoolInput(
            name="parent_node",
            display_name="Parent Node?",
            info="Turn this on if the go_to node is belong to parent graph.",
            value=False,
            advanced=True,
        )
    ]

    outputs = [
        Output(display_name="Command", name="output", method="build_output"),
    ]

    def _pre_run_setup(self):
        if self.condition_type == "Built-in":
            if self.goto_type_true == "GraphNode":
                # Convert string routes to lists
                self.true_nodes_list = [
                    node_name.strip()
                    if node_name.strip().lower() != "end"
                    else END
                    for node_name in self.true_route_go_to.split(",")]
            if self.goto_type_false == "GraphNode":
                self.false_nodes_list = [
                    node_name.strip()
                    if node_name.strip().lower() != "end" else END
                    for node_name in self.false_route_go_to.split(",")]
        elif self.condition_type == "Custom":
            # Convert goto_custom values to lists
            for k, v in self.goto_custom.items():
                # If v is a send api
                if v.startswith("{") and v.endswith("}"):
                    # Iter send_api_list and get the dict that contain send_api_func base on key which is v
                    for dict_send_api in self.send_api_custom:
                        if v.strip("{} ") in dict_send_api:
                            send_api_func_ = dict_send_api[v.strip("{} ")]
                            self.goto_custom[k] = send_api_func_
                            break
                # If v is a list of node
                else:
                    node_names = []
                    for name in v.split(","):
                        if name.strip().lower() != "end":
                            node_names.append(name.strip())
                        else:
                            node_names.append(END)
                    self.goto_custom[k] = node_names

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


    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "condition_type":
            if field_value == "Built-in":
                build_config["state_field"]["show"] = True
                build_config["state_field"]["required"] = True

                build_config["operator"]["show"] = True
                build_config["operator"]["required"] = True

                build_config["match_value"]["show"] = True
                build_config["match_value"]["required"] = True

                build_config["custom_condition"]["show"] = False
                build_config["custom_condition"]["required"] = False

                build_config["goto_type_true"]["show"] = True
                build_config["goto_type_true"]["required"] = True
                build_config["goto_type_false"]["show"] = True
                build_config["goto_type_false"]["required"] = True
                if build_config["goto_type_true"]["value"] == "GraphNode":
                    build_config["true_route_send"]["show"] = False
                    build_config["true_route_send"]["required"] = False
                    build_config["true_route_go_to"]["show"] = True
                    build_config["true_route_go_to"]["required"] = True
                elif build_config["goto_type_true"]["value"] == "Send API":
                    build_config["true_route_send"]["show"] = True
                    build_config["true_route_send"]["required"] = True
                    build_config["true_route_go_to"]["show"] = False
                    build_config["true_route_go_to"]["required"] = False
                build_config["true_route_update_value"]["show"] = True
                build_config["true_route_update_value"]["required"] = True
                if build_config["goto_type_false"]["value"] == "GraphNode":
                    build_config["false_route_send"]["show"] = False
                    build_config["false_route_send"]["required"] = False
                    build_config["false_route_go_to"]["show"] = True
                    build_config["false_route_go_to"]["required"] = True
                elif build_config["goto_type_false"]["value"] == "Send API":
                    build_config["false_route_send"]["show"] = True
                    build_config["false_route_send"]["required"] = True
                    build_config["false_route_go_to"]["show"] = False
                    build_config["false_route_go_to"]["required"] = False
                build_config["false_route_update_value"]["show"] = True
                build_config["false_route_update_value"]["required"] = True

                build_config["goto_custom"]["show"] = False
                build_config["goto_custom"]["required"] = False
                build_config["send_api_custom"]["show"] = False
                build_config["update_value"]["show"] = False
                build_config["update_value"]["required"] = False
            elif field_value == "Custom":
                build_config["state_field"]["show"] = False
                build_config["state_field"]["required"] = False

                build_config["operator"]["show"] = False
                build_config["operator"]["required"] = False

                build_config["match_value"]["show"] = False
                build_config["match_value"]["required"] = False

                build_config["custom_condition"]["show"] = True
                build_config["custom_condition"]["required"] = True

                build_config["goto_type_true"]["show"] = False
                build_config["goto_type_true"]["required"] = False
                build_config["goto_type_false"]["show"] = False
                build_config["goto_type_false"]["required"] = False
                build_config["true_route_send"]["show"] = False
                build_config["true_route_send"]["required"] = False
                build_config["true_route_go_to"]["show"] = False
                build_config["true_route_go_to"]["required"] = False
                build_config["true_route_update_value"]["show"] = False
                build_config["true_route_update_value"]["required"] = False
                build_config["false_route_send"]["show"] = False
                build_config["false_route_send"]["required"] = False
                build_config["false_route_go_to"]["show"] = False
                build_config["false_route_go_to"]["required"] = False
                build_config["false_route_update_value"]["show"] = False
                build_config["false_route_update_value"]["required"] = False

                build_config["goto_custom"]["show"] = True
                build_config["goto_custom"]["required"] = True
                build_config["send_api_custom"]["show"] = True
                build_config["update_value"]["show"] = True
                build_config["update_value"]["required"] = True

        if field_name == "goto_type_true":
            if field_value == "GraphNode":
                build_config["true_route_send"]["show"] = False
                build_config["true_route_send"]["required"] = False
                build_config["true_route_go_to"]["show"] = True
                build_config["true_route_go_to"]["required"] = True
            elif field_value == "Send API":
                build_config["true_route_send"]["show"] = True
                build_config["true_route_send"]["required"] = True
                build_config["true_route_go_to"]["show"] = False
                build_config["true_route_go_to"]["required"] = False

        if field_name == "goto_type_false":
            if field_value == "GraphNode":
                build_config["false_route_send"]["show"] = False
                build_config["false_route_send"]["required"] = False
                build_config["false_route_go_to"]["show"] = True
                build_config["false_route_go_to"]["required"] = True
            elif field_value == "Send API":
                build_config["false_route_send"]["show"] = True
                build_config["false_route_send"]["required"] = True
                build_config["false_route_go_to"]["show"] = False
                build_config["false_route_go_to"]["required"] = False
        return build_config


    def evaluate_condition(self, input_text: str | dict | list, match_text: str, operator: str, *, case_sensitive: bool) -> bool:  # noqa: E501
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

    def build_type_hint(self):
        # Determine type hint based on goto_custom keys
        type_hint_list = []
        if self.condition_type == "Built-in":
            # Determine type hint based on the number of nodes in true and false route
            if self.goto_type_true == "GraphNode":
                if len(self.true_nodes_list) == 1:
                    type_hint_list.append(self.true_nodes_list[0])
                elif len(self.true_nodes_list) > 1 and "Sequence[str]" not in type_hint_list:
                    type_hint_list.append("Sequence[str]")
            elif self.goto_type_true == "Send API":
                type_hint_list.append("Sequence[Send]")

            if self.goto_type_false == "GraphNode":
                if len(self.false_nodes_list) == 1:
                    type_hint_list.append(self.false_nodes_list[0])
                elif len(self.false_nodes_list) > 1 and "Sequence[str]" not in type_hint_list:
                    type_hint_list.append("Sequence[str]")
            elif self.goto_type_false == "Send API":
                type_hint_list.append("Sequence[Send]")


        elif self.condition_type == "Custom":
            for v in self.goto_custom.values():
                # If v is Send API
                if isinstance(v, Callable) and "Sequence[Send]" not in type_hint_list:
                    type_hint_list.append("Sequence[Send]")
                else:
                    if len(v) == 1:
                        type_hint_list.append(v[0])
                    if len(v) > 1 and "Sequence[str]" not in type_hint_list:
                        type_hint_list.append("Sequence[str]")

        # Form type hint
        if len(type_hint_list) == 1 and type_hint_list[0] == "Sequence[str]":
            type_hint = Command[Sequence[str]]
        elif len(type_hint_list) == 1 and type_hint_list[0] == "Sequence[Send]":
            type_hint = Command[Sequence[Send]]
        else:
            type_hint = eval(f"Command[Literal[{', '.join(repr(v) for v in type_hint_list)}]]")  # noqa: S307
        print(f"Type hint for conditional edge: {type_hint}")  # noqa: T201
        return type_hint

    def _safe_literal_eval(self, value: str, expected_type=None):
        """Safely evaluate literal with type checking."""
        import ast
        result = ast.literal_eval(value)
        print(f"Evaluated value: {result} of type {type(result)}")  # noqa: T201
        if expected_type and not isinstance(result, expected_type):
            msg = f"Expected {expected_type.__name__}, got {type(result).__name__}"
            raise ValueError(msg)
        return result


    def build_output(self) -> "CommandAddonForLangGraph":

        self.type_hint = self.build_type_hint()
        def command_node(result, state) -> self.type_hint: # type: ignore  # noqa: PGH003
            if self.condition_type == "Built-in":
                # Extract the state field
                compared_value = getattr(result, self.state_field) if hasattr(result, self.state_field) else result[self.state_field]  # noqa: E501
                evaluated_result = self.evaluate_condition(compared_value, self.match_value, self.operator, case_sensitive=self.case_sensitive)  # noqa: E501
                if evaluated_result:
                    update_value = self.normalize_input_data(result, self.true_route_update_value)
                    if self.goto_type_true == "GraphNode":
                        goto = self.true_nodes_list
                    elif self.goto_type_true == "Send API":
                        # Get the first and only send API from true_route_send
                        goto = self.true_route_send[next(iter(self.true_route_send))](state, update_value)
                else:
                    update_value = self.normalize_input_data(result, self.false_route_update_value)
                    if self.goto_type_false == "GraphNode":
                        goto = self.false_nodes_list
                    elif self.goto_type_false == "Send API":
                        # Get the first and only send API from false_route_send
                        goto = self.false_route_send[next(iter(self.false_route_send))](state, update_value)
            elif self.condition_type == "Custom":
                evaluated_result = self.custom_condition(result)
                goto_ = self.goto_custom.get(evaluated_result)
                update_value = self.update_value.get(evaluated_result)
                update_value = self.normalize_input_data(result, update_value)
                if isinstance(goto_, Callable):  # noqa: SIM108
                    # If goto is a callalbe which, pass in state to create Send API
                    goto = goto_(state, update_value)
                else:
                    # If it is normal node, just use it
                    goto = goto_

            return Command(
                update=update_value,
                goto=goto,
                graph=Command.PARENT if self.parent_node else None
            )

        print(f"Building output with type hint: {self.type_hint}")  # noqa: T201
        self.function_ = command_node
        return self # This will print the type of the first key in the dictionary


    ############## Input Normalization and Custom Model Handling ##############
    def normalize_input_data(self, dict_or_object=None, input_data=None):
        """Normalize input data based on type flags.

        dict_or_object: This will be used to support @is_field. Get the field in dict_or_object.
        Could be things like node output, state, etc.

        input_data: A dictionary which key is normal string and value is a declaration
        string that need to be converted to the correct type.
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

            except (ValueError, KeyError, AttributeError, TypeError) as e:
                print(f"✗ Error converting {key}: {e} || Check your prompt again. Keeping original value.")  # noqa: T201
                input_data[key] = original_value
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
        for _i, declare in enumerate(declarations):
            d = declare.strip()

            if d.startswith("@is_class"):
                clean_value = d.replace("@is_class", "").strip()

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

            if d.startswith("@is_field"):
                if not dict_or_object:
                    msg = "No dictionary or object provided for @is_field. Check your code."
                    raise ValueError(msg)
                clean_value = d.replace("@is_field", "").strip()
                if " as " in clean_value:
                    node_output_value_field, var_name = clean_value.split(" as ", 1)
                    var_name = var_name.strip()
                    node_output_value = self._parse_field(dict_or_object, node_output_value_field.strip())
                    variables[var_name] = node_output_value
                    print(f"Created variable '{var_name}': {node_output_value}")  # noqa: T201
                else:
                    msg = "Node output declaration must include 'as variable_name' part."
                    raise ValueError(msg)

            elif d.startswith("@is_list"):
                clean_value = d.replace("@is_list", "").strip()
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

            elif d.startswith("@is_dict"):
                clean_value = d.replace("@is_dict", "").strip()
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
