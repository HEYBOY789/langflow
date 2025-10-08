from typing import Any  # noqa: N999

from src.backend.base.langflow.components.langflow.utils.graph_node_func import (
    build_params_for_add_node,
    detect_and_register_edges,
)

from langflow.custom import Component
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output


class GraphNodeAsSubGraph(Component):
    display_name = "Graph Node As Sub Graph"
    description = "Connect a Graph to this component to use it as a subgraph, "
    "with this node acting as its representation in the parent graph"
    documentation: str = "https://langchain-ai.github.io/langgraph/concepts/subgraphs/"
    icon = "LangChain"
    name = "LangGraphNodeAsSubGraph"

    inputs = [
        HandleInput(
            name="previous_nodes",
            display_name="Previous Nodes",
            info="Connect previous GraphNode(s) to create edges. "
            "Can connect multiple nodes for many-to-one relationships. "
            "If State Graph is connected instead of previous nodes, "
            "this node will be treated as the START node in the graph.",
            input_types=[
                "GraphNodeAsSubGraph",
                "GraphNodeForCrewAIAgent",
                "GraphNodeForAgent",
                "GraphNodeForCrewAICrew",
                "GraphNodeForFunction",
                "ConditionalEdgeForLangGraph",
                "CreateStateGraphComponent",
                "SendMapReduceForLangGraph",
                "GraphNodeForAgentWithCommand",
                "GraphNodeForCrewAIAgentWithCommand",
                "GraphNodeForCrewAICrewWithCommand",
                "GraphNodeForFunctionWithCommand",
                "GraphNodeAsSubGraphWithCommand"
                ],
            is_list=True,
            required=True
        ),
        MessageTextInput(
            name="node_name",
            display_name="Node Name",
            info="Name of the node in the graph. This will be used to identify the node in the graph.",
            placeholder="Example: single_node",
            required=True,
        ),
        HandleInput(
            name="sub_graph",
            input_types=["GraphRun"],
            display_name="Sub Graph",
            info="Connect a StateGraph (Graph Runner) to use it as a subgraph.",
            required=True,
        ),
        HandleInput(
            name="input_state",
            display_name="Input State Of Node.",
            info="Input state of the function. "
            "If not provided, graph input state will be used by taken from the first previous node.",
            input_types=["ModelClassWrapper"],
        ),
        HandleInput(
            name="output_state",
            display_name="Output State Of Node.",
            info="Output state of the function. "
            "If not provided, graph output state will be used by taken from the first previous node.",
            input_types=["ModelClassWrapper"],
        ),
        BoolInput(
            name="differ_state",
            display_name="Differ State Schemas?",
            value=False,
            info="Turn this on if there is no shared state keys in parent and subgraph schemas",
            real_time_refresh=True
        ),
        MessageTextInput(
            name="subgraph_extract_field",
            display_name="Subgraph Extract Field",
            info="Enter the field name that you want to extract value from subgraph output state. "
            "If leave empty, the whole subgraph output state will be returned.",
            placeholder="Example: foo",
            show=False,
            required=False,
            dynamic=True
        ),
        MessageTextInput(
            name="output_state_field",
            display_name="Output State Field",
            info="Enter the field name where the subgraph output value should be stored in this node output state",
            placeholder="Example: bar",
            show=False,
            required=False,
            dynamic=True
        ),
        BoolInput(
            name="command_to_parent",
            display_name="Command To Parent?",
            info="Enable this option if any node in the subgraph returns "
            "a Command that directs to a node in the parent graph."
            "The Command must always point to a parent node in every evaluation. "
            "You cannot mix behaviors—for example, "
            "returning a node within the same subgraph in some cases and a parent node in others.",
            value=False,
            advanced=True,
        ),
        IntInput(
            name="node_caching",
            display_name="Node Caching Time",
            info="Enable caching for this node. Count by second. Make sure to enable node caching in Graph Runner. (https://langchain-ai.github.io/langgraph/concepts/low_level/#node-caching).",
            advanced=True,
            value=0
        ),
        HandleInput(
            name="retry_policy",
            display_name="Retry Policy",
            info="Connect a Retry Policy Function component to handle retries for this node.",
            input_types=["CreateRetryPolicyFunctionForLangGraph"],
            advanced=True
        ),
        BoolInput(
            name="defer_node",
            display_name="Defer Node?",
            info="Deferring node execution is useful when you want to delay "
            "the execution of a node until all other pending tasks are completed. "
            "This is particularly relevant when branches have different lengths, "
            "which is common in workflows like map-reduce flows. "
            "(https://langchain-ai.github.io/langgraph/how-tos/graph-api/#defer-node-execution)",
            advanced=True,
            value=False,
        )
    ]

    outputs = [
        Output(display_name="Sub Graph Node", name="next_node", method="build_graph")
    ]

    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "differ_state":
            if field_value:
                build_config["subgraph_extract_field"]["show"] = True
                build_config["subgraph_extract_field"]["required"] = True
                build_config["output_state_field"]["show"] = True
                build_config["output_state_field"]["required"] = True
            else:
                build_config["subgraph_extract_field"]["show"] = False
                build_config["subgraph_extract_field"]["required"] = False
                build_config["output_state_field"]["show"] = False
                build_config["output_state_field"]["required"] = False
        return build_config


    def _update_class_identity(self):
        """Update the class identity based on whether command addon is connected."""
        if self.command_to_parent:
            # Change the class name for detection by other components
            self.__class__.__name__ = "GraphNodeAsSubGraphWithCommand"
            self.name = "LangGraphNodeAsSubGraphWithCommand"
            self.display_name = "Graph Node As Sub Graph With Command"
        else:
            # Reset to original class name
            self.__class__.__name__ = "GraphNodeAsSubGraph"
            self.name = "LangGraphNodeAsSubGraph"
            self.display_name = "Graph Node As Sub Graph"


    def _pre_run_setup(self):
        # Update class identity before setup
        self._update_class_identity()

        self.graph_output_state = self.previous_nodes[0].graph_output_state
        self.graph_input_state = self.previous_nodes[0].graph_input_state
        self.graph_builder = self.previous_nodes[0].graph_builder
        self.graph_context_schema = self.previous_nodes[0].graph_context_schema

        # If output_state is not provied, get from shared data
        if not self.output_state:
            self.output_state = self.graph_output_state
        self.output_model = self.output_state.model_class

         # If input_state is not provided, get from shared data
        if not self.input_state:
            self.input_state = self.graph_input_state
        self.input_model = self.input_state.model_class


    def run_subgraph(self, state) -> dict[str, Any]:
        try:
            # Pass in parent state to normalize input data
            self.sub_graph.parent_state = state
            result = self.sub_graph._execute_graph_async_subgraph()
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            msg = f"{self.node_name} | Exception: {e}"
            raise ValueError(msg) from e

        if self.subgraph_extract_field:
            if isinstance(result, dict):
                if self.subgraph_extract_field in result:
                    result = result[self.subgraph_extract_field]
                else:
                    msg = (
                        f"{self.node_name} | "
                        f"Field '{self.subgraph_extract_field}' not found in subgraph output: {result}"
                        )
                    raise ValueError(msg)
            elif hasattr(result, self.subgraph_extract_field):
                # If result is an object, get the attribute
                result = getattr(result, self.subgraph_extract_field)
            else:
                msg = f"{self.node_name} | Field '{self.subgraph_extract_field}' not found in subgraph output: {result}"
                raise ValueError(msg)
        # result is already set, no need to reassign
        return {self.output_state_field: result}


    # Add this node to builder
    def build_graph(self) -> "GraphNodeAsSubGraph": # type: ignore  # noqa: PGH003
        # Get the shared builder from context
        builder = self.graph_builder
        if not builder:
            msg = (
                f"{self.node_name} | No StateGraph builder found in context. "
                "Make sure CreateStateGraph component is connected and executed first."
            )
            raise ValueError(msg)

        # Define node function with proper type hints
        def node_function(state: self.input_model) -> self.output_model: # type: ignore  # noqa: PGH003
            try:
                result = self.run_subgraph(state)
                print(f"Node {self.node_name} returning sub-graph result: {result}")  # noqa: T201
            except (ValueError, TypeError, AttributeError, KeyError) as e:
                msg = f"{self.node_name} | Exception: {e}"
                raise ValueError(msg) from e
            return result

        params = build_params_for_add_node(self.node_caching, self.retry_policy, self.defer_node)

        if self.differ_state:
            builder.add_node(self.node_name, node_function, **params)
        else:
            builder.add_node(self.node_name, self.sub_graph._graph, **params)

        print(f"Added node: {self.node_name}")  # noqa: T201

        # THEN detect and register edges (after node exists)
        detect_and_register_edges(builder, self.node_name, self.previous_nodes)
        return self
