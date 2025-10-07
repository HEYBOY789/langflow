import re  # noqa: N999
from collections.abc import Callable, Sequence

from langgraph.graph import END, START

from langflow.custom.custom_component.component import Component
from langflow.io import BoolInput, DictInput, DropdownInput, HandleInput, MessageTextInput, Output


class ConditionalEdgeForLangGraph(Component):
    display_name = "Conditional Edge For LangGraph"
    description = "Use this component to create a conditional edge in LangGraph."
    documentation: str = "https://langchain-ai.github.io/langgraph/how-tos/graph-api/#conditional-branching"
    icon = "LangChain"
    name = "ConditionalEdgeForLangGraph"

    inputs = [
        HandleInput(
            name="previous_node",
            display_name="Previous Node",
            info="Connect previous GraphNode to create conditional edge. "
            "If State Graph is connected instead of GraphNode, this will be treated as the Conditional Entry Point.",
            input_types=[
                "GraphNodeAsSubGraph",
                "GraphNodeForCrewAIAgent",
                "GraphNodeForAgent",
                "GraphNodeForCrewAICrew",
                "GraphNodeForFunction",
                "CreateStateGraphComponent"
                ],
            required=True,
            show=True,
            dynamic=True
        ),
        HandleInput(
            name="input_state",
            display_name="Input State Of The Conditional Edge",
            info="Input state of the conditional edge. "
            "If not provided, graph input state will be used by taken from the previous node.",
            input_types=["ModelClassWrapper"],
        ),
        MessageTextInput(
            name="conditional_edge_name",
            display_name="Name Of The Conditional Edge",
            info="Name of the conditional edge",
            required=True,
        ),
        DropdownInput(
            name="condition_type",
            display_name="Condition Type",
            info="Choose either built-in or custom condition type. "
            "If you only need to check 1 value and there is a suitable operator, use built-in condition type. "
            "If you need to check multiple values or use custom logic, use custom condition type.",
            options=["Built-in", "Custom"],
            required=True,
            value="Built-in",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="state_field",
            display_name="Field Name Of The Input State",
            info="Specify the field name of the input state you want to use for the condition. "
            "Beside string and number, you can also check list and dict. "
            "For a list, you can check its length or if it contains a specific value. "
            "For a dict, you can check its length or if it contains a specific key.",
            required=True,
            dynamic=True,
            show=True
        ),
        DropdownInput(
            name="operator",
            display_name="Operator",
            info="Choose an operator to compare the input state value with the condition value.",
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
        BoolInput(
            name="case_sensitive",
            display_name="Case Sensitive",
            info="If true, the comparison will be case sensitive.",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="true_route",
            display_name="True Route",
            info="Specify the node names to route to if the condition is true. "
            "Use 'END' to route to the end of the graph. "
            "If using multiple nodes, use commas to separate them. Example: node1, node2, end",
            required=True,
            dynamic=True,
            show=True
        ),
        MessageTextInput(
            name="false_route",
            display_name="False Route",
            info="Specify the node names to route to if the condition is false. "
            "Use 'END' to route to the end of the graph. "
            "If using multiple nodes, use commas to separate them. Example: node1, node2, end",
            required=True,
            dynamic=True,
            show=True
        ),
        HandleInput(
            name="custom_condition",
            input_types=["Callable"],
            display_name="Function",
            info=(
                "Connect a Python function to use its custom logic. Must return var_name. "
                "Must have one input reference to input_state. Access value of fields in state by using state.field.\n"
                "Example:\n"
                "def conditional_edge(state):\n"
                "____if state.field == 'condi_1':\n"
                "________return 'var_name_1'\n"
                "____elif state.field == 'condi_2':\n"
                "________return 'var_name_2'\n"
                "____else:\n"
                "________return 'var_name_3'\n"
            ),
            required=False,
            show=False,
            dynamic=True,
        ),
        DictInput(
            name="dict_route",
            display_name="Routing",
            info="Specify the var_name and the node_name. "
            "The custom code must return the var_name and it will be referred to the node_name. Use 'end' to refer END."
            " If using multiple nodes, use commas to separate them. Example: node1, node2, end",
            required=False,
            dynamic=True,
            show=False,
            value={"var_name_1": "node_name_1", "var_name_2": "node_name_2", "var_name_3": "node_name_3"},
            is_list=True,
        )
    ]

    outputs = [
        Output(
            display_name="Condition Edges (visualization only)",
            name="build_conditional_edge",
            method="build_function",
            info="This is just for visualization purposes, it does not actually build a route. "
            "But you must connect this to other nodes in the route to complete the flow."
        ),
    ]

    def _pre_run_setup(self):
        self.graph_input_state = self.previous_node.graph_input_state
        self.graph_builder = self.previous_node.graph_builder
        self.graph_output_state = self.previous_node.graph_output_state
        self.graph_context_schema = self.previous_node.graph_context_schema

        if self.condition_type == "Built-in":
            # Convert string routes to lists
            self.true_nodes_list = [
                node_name.strip()
                if node_name.strip().lower() != "end" else END
                for node_name in self.true_route.split(",")
                ]
            self.false_nodes_list = [
                node_name.strip() if node_name.strip().lower() != "end" else END
                for node_name in self.false_route.split(",")
            ]
        elif self.condition_type == "Custom":
            # Convert dict_route values to lists
            for k, v in self.dict_route.items():
                self.dict_route[k] = [
                    name.strip() if name.strip().lower() != "end" else END
                    for name in v.split(",")
                ]

        # If input_state is not provided, get from shared data
        if not self.input_state:
            self.input_state = self.graph_input_state
        self.input_model = self.input_state.model_class


    def _detect_and_register_edges(self, conditional_func: Callable):
        """Detect previous_nodes connections and add them to conditional edge."""
        builder = self.graph_builder
        if not builder:
            msg = f"{self.conditional_edge_name} | No StateGraph builder found in context."
            raise ValueError(msg)

        if self.previous_node:
            # Create conditional entry point if previous node is from CreateStateGraphComponent
            if self.previous_node.__class__.__name__ == "CreateStateGraphComponent":
                builder.add_conditional_edges(START, conditional_func)
                print(f"Added conditional edge: START -> {conditional_func.__name__}")  # noqa: T201
            else:
                # Get the node name from the previous GraphNode component
                prev_node_name = self.previous_node.node_name
                builder.add_conditional_edges(prev_node_name, conditional_func)
                print(f"Added conditional edge: {prev_node_name} -> {conditional_func.__name__}")  # noqa: T201


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

                build_config["true_route"]["show"] = True
                build_config["true_route"]["required"] = True
                build_config["false_route"]["show"] = True
                build_config["false_route"]["required"] = True

                build_config["dict_route"]["show"] = False
                build_config["dict_route"]["required"] = False
            elif field_value == "Custom":
                build_config["state_field"]["show"] = False
                build_config["state_field"]["required"] = False

                build_config["operator"]["show"] = False
                build_config["operator"]["required"] = False

                build_config["match_value"]["show"] = False
                build_config["match_value"]["required"] = False

                build_config["custom_condition"]["show"] = True
                build_config["custom_condition"]["required"] = True

                build_config["true_route"]["show"] = False
                build_config["true_route"]["equired"] = False
                build_config["false_route"]["show"] = False
                build_config["false_route"]["equired"] = False

                build_config["dict_route"]["show"] = True
                build_config["dict_route"]["required"] = True
        return build_config


    def evaluate_condition(self,
                           input_text: str | dict | list,
                           match_text: str,
                           operator: str,
                           *,
                           case_sensitive: bool) -> bool:
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

            elif operator == "equals":
                # Check if list length equals match_text as number
                try:
                    return len(input_text) == float(match_text)
                except ValueError:
                    return False

            elif operator == "not equals":
                # Check if list length does not equal match_text as number
                try:
                    return len(input_text) != float(match_text)
                except ValueError:
                    return False

            else:
                return False  # Unsupported operator for list

        # Handle Dict operations
        elif isinstance(input_text, dict):
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

            elif operator == "equals":
                # Check if dict length equals match_text as number
                try:
                    return len(input_text) == float(match_text)
                except ValueError:
                    return False

            elif operator == "not equals":
                # Check if dict length does not equal match_text as number
                try:
                    return len(input_text) != float(match_text)
                except ValueError:
                    return False

            else:
                return False  # Unsupported operator for dict

        # Handle String operations (existing logic)
        else:
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
            elif operator in ["less than", "less than or equal", "greater than", "greater than or equal"]:
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
        # Determine type hint based on dict_route keys
        type_hint_list = []
        if self.condition_type == "Built-in":
            # Determine type hint based on the number of nodes in true and false routes
            if len(self.true_nodes_list) == 1:
                type_hint_list.append(self.true_nodes_list[0])
            elif len(self.true_nodes_list) > 1 and "Sequence[str]" not in type_hint_list:
                type_hint_list.append("Sequence[str]")

            if len(self.false_nodes_list) == 1:
                type_hint_list.append(self.false_nodes_list[0])
            elif len(self.false_nodes_list) > 1 and "Sequence[str]" not in type_hint_list:
                type_hint_list.append("Sequence[str]")

        elif self.condition_type == "Custom":
            for v in self.dict_route.values():
                if len(v) == 1:
                    type_hint_list.append(v[0])
                elif len(v) > 1 and "Sequence[str]" not in type_hint_list:
                    type_hint_list.append("Sequence[str]")

        # Form type hint
        if len(type_hint_list) == 1 and type_hint_list[0] == "Sequence[str]":
            type_hint = Sequence[str]
        else:
            type_hint = eval(f"Literal[{', '.join(repr(v) for v in type_hint_list)}]")  # noqa: S307
        return type_hint


    def build_function(self) -> "ConditionalEdgeForLangGraph":
        type_hint = self.build_type_hint()
        def conditional_edge(state: self.input_model) -> type_hint: # type: ignore  # noqa: PGH003
            if self.condition_type == "Built-in":
                result = self.evaluate_condition(
                    getattr(state, self.state_field, ""),
                    self.match_value,
                    self.operator,
                    case_sensitive=self.case_sensitive
                    )
                if result:
                    return self.true_nodes_list
                return self.false_nodes_list
            if self.condition_type == "Custom":
                result = self.custom_condition(state)
                print(f"Custom condition evaluated: {result}")  # noqa: T201
                return self.dict_route.get(result)
            return None

        print(f"Building conditional edge: {self.conditional_edge_name}")  # noqa: T201
        conditional_edge.__name__ = self.conditional_edge_name
        conditional_edge.__qualname__   = self.conditional_edge_name

        # Register the conditional edge with the builder
        self._detect_and_register_edges(conditional_edge)

        self.build_function_run = True

        return self
