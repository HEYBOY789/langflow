from collections.abc import Callable, Sequence  # noqa: N999
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, Send
from src.backend.base.langflow.components.LangGraph.utils.conditional_func import evaluate_condition
from src.backend.base.langflow.components.LangGraph.utils.input_convert_func import normalize_input_data

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
        elif len(type_hint_list) == 1:
            type_hint = Command[type_hint_list[0]]
        else:
            # Create type hint with literal values directly without using eval
            from typing import _LiteralGenericAlias
            literal_type = _LiteralGenericAlias(Literal, tuple(type_hint_list))
            type_hint = Command[literal_type]
        print(f"Type hint for conditional edge: {type_hint}")  # noqa: T201
        return type_hint


    def build_output(self) -> "CommandAddonForLangGraph":

        self.type_hint = self.build_type_hint()
        def command_node(result, state) -> self.type_hint: # type: ignore  # noqa: PGH003
            if self.condition_type == "Built-in":
                # Extract the state field
                compared_value = getattr(result, self.state_field) if hasattr(result, self.state_field) else result[self.state_field]  # noqa: E501
                evaluated_result = evaluate_condition(compared_value, self.match_value, self.operator, case_sensitive=self.case_sensitive)  # noqa: E501
                if evaluated_result:
                    update_value = normalize_input_data(self.true_route_update_value, result)
                    if self.goto_type_true == "GraphNode":
                        goto = self.true_nodes_list
                    elif self.goto_type_true == "Send API":
                        # Get the first and only send API from true_route_send
                        goto = self.true_route_send[next(iter(self.true_route_send))](state, update_value)
                else:
                    update_value = normalize_input_data(self.false_route_update_value, result)
                    if self.goto_type_false == "GraphNode":
                        goto = self.false_nodes_list
                    elif self.goto_type_false == "Send API":
                        # Get the first and only send API from false_route_send
                        goto = self.false_route_send[next(iter(self.false_route_send))](state, update_value)
            elif self.condition_type == "Custom":
                evaluated_result = self.custom_condition(result)
                goto_ = self.goto_custom.get(evaluated_result)
                update_value = self.update_value.get(evaluated_result)
                update_value = normalize_input_data(update_value, result)
                print(f"Update value after processing: {update_value}")  # noqa: T201
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
