import json  # noqa: N999
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from src.backend.base.langflow.components.langflow.utils.agent_result_func import clarify_result
from src.backend.base.langflow.components.langflow.utils.graph_node_func import (
    build_params_for_add_node,
    check_if_field_is_list,
    detect_and_register_edges,
)
from src.backend.base.langflow.components.langflow.utils.memory_func import extract_memory, store_memory
from src.backend.base.langflow.components.langflow.utils.prompt_func import (
    form_memory_str_for_prompt,
    format_all_prompts,
)

from langflow.custom import Component
from langflow.io import BoolInput, DropdownInput, HandleInput, IntInput, MessageTextInput, MultilineInput, Output
from langflow.schema.dotdict import dotdict


class GraphNodeForCrewAIAgent(Component):
    display_name = "Graph Node For CrewAI Agent"
    description = "Node for LangGraph that processes input with CrewAI agents component."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "LangChain"
    name = "LangGraphNodeForCrewAIAgent"

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
            name="crewai_agent",
            display_name="CrewAI Agent",
            info="Connect to CrewAI Agent",
            input_types=["Agent"],
            required=True
        ),
        MultilineInput(
            name="prompt",
            display_name="User Prompt",
            info="Prompt to be sent to the agent.\n\n"
            "• Prompt must be a message.\n"
            "• Use double brackets to reference inputs, e.g. {input_name}.\n"
            "• Variables in the prompt must match the input state fields.\n"
            "• If using Pydantic, make sure you instruct "
            "the agent to return a json object that matches the output model schema.\n"
            'Example: Only return a json object with the following structure: {"field_name": "value"}. Nothing more.\n',
            placeholder="Example: Please analyze the following text: {content} based on the model's "
            "verbose instructions:{langflow_model_schema}.",
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
            "If not provided, graph output state will be used by taken from the first previous node."
            "Use syntax {langflow_model_schema} to include the model's verbose schema in User Prompt and CrewAI Agent Prompts.",  # noqa: E501
            input_types=["ModelClassWrapper"],
        ),
        DropdownInput(
            name="output_type",
            display_name="Output Pydantic or a State Field",
            info="Select the output type for this node. If you select Pydantic, the output will be a Pydantic model. "
            "If you select State Field, you can assign the output (single value) to a state field by manually "
            "typing the field name.",
            options=["Pydantic", "State Field"],
            value="Pydantic",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="output_state_field",
            display_name="Output State Field",
            info="If you choose 'State Field' as the output type, "
            "enter the field name where the output should be stored. If an output state model is defined for the node, "
            "the field name must match one of its fields. By default, the graph's output state will be used."
            "If choosing 'State Field' as the output type, use {langflow_model_schema} in agent's prompt to include "
            "the model's verbose instructions.",
            placeholder="Example: result",
            show=False,
            required=False,
            dynamic=True,
        ),
        HandleInput(
            name="return_command_addon",
            display_name="Command Addon",
            input_types=["CommandAddonForLangGraph"],
            info="Connect a Command Addon to make this node return a Command object.",
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
            info="Deferring node execution is useful when you want to delay the execution of a node until all other "
            "pending tasks are completed. This is particularly relevant when branches have different lengths, "
            "which is common in workflows like map-reduce flows. "
            "(https://langchain-ai.github.io/langgraph/how-tos/graph-api/#defer-node-execution)",
            advanced=True,
            value=False,
        ),
        BoolInput(
            name="use_mem",
            display_name="Use Long-Term Memory?",
            value=False,
            info="Enable long-term memory for this node.",
            real_time_refresh=True
        ),
        HandleInput(
            name="get_from_mem_addon",
            input_types=["GetLongMemAddonForGraphNode"],
            display_name="Get Long Memory Addon",
            info="Connect a Get Long Memory Addon components to enable retrieving long-term memory for this node.",
            is_list=True,
            show=False,
            dynamic=True
        ),
        HandleInput(
            name="put_to_mem_addon",
            input_types=["StoreLongMemAddonForGraphNode"],
            display_name="Store Long Memory Addon",
            info="Connect a Store Long Memory Addon components to enable storing long-term memory for this node.",
            is_list=True,
            dynamic=True,
            show=False,
        )
    ]

    outputs = [
        Output(display_name="CrewAI Agent Node", name="next_node", method="build_graph")
    ]

    def _update_class_identity(self):
        """Update the class identity based on whether command addon is connected."""
        if hasattr(self, "return_command_addon") and self.return_command_addon:
            # Change the class name for detection by other components
            self.__class__.__name__ = "GraphNodeForCrewAIAgentWithCommand"
            self.name = "LangGraphNodeForCrewAIAgentWithCommand"
            self.display_name = "Graph Node For CrewAI Agent With Command"
        else:
            # Reset to original class name
            self.__class__.__name__ = "GraphNodeForCrewAIAgent"
            self.name = "LangGraphNodeForCrewAIAgent"
            self.display_name = "Graph Node For CrewAI Agent"


    def _pre_run_setup(self):
        # Update class identity before setup
        self._update_class_identity()

        # Cache memories locally to avoid multiple calls in one run
        self.memories = []

        self.graph_output_state = self.previous_nodes[0].graph_output_state
        self.graph_input_state = self.previous_nodes[0].graph_input_state
        self.graph_builder = self.previous_nodes[0].graph_builder
        self.graph_context_schema = self.previous_nodes[0].graph_context_schema

        if self.graph_context_schema:
            from langgraph.runtime import Runtime
            self.context_schema_model = self.graph_context_schema.model_class
            self.graph_runtime = Runtime[self.context_schema_model]
        else:
            self.graph_runtime = None

        # If output_state is not provied, get from shared data
        if not self.output_state:
            self.output_state = self.graph_output_state
        self.output_model = self.output_state.model_class

        # If input_state is not provided, get from shared data
        if not self.input_state:
            self.input_state = self.graph_input_state
        self.input_model = self.input_state.model_class


    async def run_crewai_agent(self, state, _original_prompts, runtime) -> dict[str, Any]:
        # Create local copies to avoid cross-contamination between parallel executions
        role, goal, backstory, prompt = _original_prompts
        mem_str = form_memory_str_for_prompt(self.memories) if self.memories else ""

        # Format role
        formatted_role = format_all_prompts(role, mem_str, runtime, state, self.node_name)

        # Format goal
        formatted_goal = format_all_prompts(goal, mem_str, runtime, state, self.node_name)

        # Format backstory
        formatted_backstory = format_all_prompts(backstory, mem_str, runtime, state, self.node_name)

        # Format prompt
        formatted_prompt = format_all_prompts(prompt, mem_str, runtime, state, self.node_name)

        # Formated agent
        self.crewai_agent.role = formatted_role
        self.crewai_agent.goal = formatted_goal
        self.crewai_agent.backstory = formatted_backstory

        print("#############################")  # noqa: T201
        print(f"Running CrewAI Agent with prompt: {formatted_prompt}") # noqa: T201
        print("#############################") # noqa: T201

        if self.output_type == "Pydantic":
            # Use the agent to process the formatted prompt
            result = await self.crewai_agent.kickoff_async(formatted_prompt, response_format=self.output_model)
            print("#############################") # noqa: T201
            print(f"Debug response from Agent: {result}") # noqa: T201
            print("#############################") # noqa: T201
            return clarify_result(result)
        # Check for output state field if output type is State Field
        available_fields = [f["name"] for f in self.output_state.schema["fields"]]
        if self.output_state_field not in available_fields:
            msg = (f"Field '{self.output_state_field}' not found in output schema. "
                   f"Available fields are: {available_fields}")
            raise ValueError(msg)

        result = await self.crewai_agent.kickoff_async(formatted_prompt)
        print(f"Debug response from Agent: {result}")  # noqa: T201

        # Check the field type from the schema
        is_list_field = check_if_field_is_list(self.output_state, self.output_state_field)
        if is_list_field:
            return {self.output_state_field: [result.raw]}
        return {self.output_state_field: result.raw}


    # Add this node to builder
    def build_graph(self) -> "GraphNodeForCrewAIAgent":
        # Get the shared builder from context
        builder = self.graph_builder
        if not builder:
            msg = (
                "No StateGraph builder found in context. "
                "Make sure CreateStateGraph component is connected and executed first."
            )
            raise ValueError(msg)

        command_type_hint = self.return_command_addon.type_hint if self.return_command_addon else None

        original_prompts = (
            self.crewai_agent.role,
            self.crewai_agent.goal,
            self.crewai_agent.backstory,
            self.prompt
        )

        # Define node function with proper type hints
        async def node_function(  # noqa: D417
                state: self.input_model, # type: ignore  # noqa: PGH003
                store: BaseStore | None=None,
                config:RunnableConfig | None=None,
                runtime: self.graph_runtime = None # type: ignore  # noqa: PGH003
                ) -> Literal[self.output_model, command_type_hint]: # type: ignore  # noqa: PGH003
            """Process input state with agent/crew and return output state.

            Args:
                state: Input state (either as dict or Pydantic model)

            Returns:
                Output state according to the output_model schema
            """
            try:
                # Try to get memories
                await extract_memory(self.get_from_mem_addon, self.memories, store, config)
                # Get result
                result = await self.run_crewai_agent(state, original_prompts, runtime)
                # Store result as long memory
                await store_memory(self.put_to_mem_addon, result, store, config)
                # Return Command object if command_addon is provided
                if self.return_command_addon:
                    # Convert result to object to synchonize when using .field
                    if isinstance(result, dict):
                        result = self.output_model(**result)
                    result = self.return_command_addon.function_(result, state)
                    print(f"Node {self.node_name} returning Command object: {result}")  # noqa: T201
                else:
                    print(f"Node {self.node_name} returning regular result: {result}")  # noqa: T201
            except (ValueError, TypeError, KeyError, json.JSONDecodeError, AttributeError) as e:
                msg = f"{self.node_name} | You may want to try another llm model to get the message | Exception: {e}"
                raise ValueError(msg) from e
            return result

        params = build_params_for_add_node(self.node_caching, self.retry_policy, self.defer_node)

        builder.add_node(
            self.node_name, node_function, **params
        )
        print(f"Added node: {self.node_name}")  # noqa: T201

        # THEN detect and register edges (after node exists)
        detect_and_register_edges(builder=builder, node_name=self.node_name, previous_nodes=self.previous_nodes)
        return self

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "output_type":
            if field_value == "Pydantic":
                build_config["output_state_field"]["show"] = False
                build_config["output_state_field"]["required"] = False
            elif field_value == "State Field":
                build_config["output_state_field"]["show"] = True
                build_config["output_state_field"]["required"] = True

        if field_name == "use_mem":
            if field_value:
                build_config["get_from_mem_addon"]["show"] = True
                build_config["put_to_mem_addon"]["show"] = True
            else:
                build_config["get_from_mem_addon"]["show"] = False
                build_config["put_to_mem_addon"]["show"] = False
        return build_config
