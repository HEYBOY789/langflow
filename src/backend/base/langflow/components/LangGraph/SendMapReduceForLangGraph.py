from typing import Any  # noqa: N999

from langgraph.types import Send
from pydantic import Field, create_model
from src.backend.base.langflow.components.LangGraph.utils.conditional_func import detect_and_register_cond_edges
from src.backend.base.langflow.components.LangGraph.utils.input_convert_func import normalize_input_data

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
                    self.extra_input = normalize_input_data(self.extra_input, state)
                    self.extra_input = {**update_value_from_command, **self.extra_input}
                else:
                    self.extra_input = update_value_from_command
            elif self.extra_input:
                self.extra_input = normalize_input_data(self.extra_input, state)

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
        self.graph_builder = detect_and_register_cond_edges(
            self.graph_builder,
            self.send_api_name,
            self.previous_node,
            send_api_edge,
            self.send_node_name
        )
        return self
