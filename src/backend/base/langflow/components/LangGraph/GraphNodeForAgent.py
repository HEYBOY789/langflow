from typing import Any, Literal  # noqa: N999

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from lfx.base.models.chat_result import get_chat_result
from src.backend.base.langflow.components.LangGraph.utils.graph_node_func import (
    build_params_for_add_node,
    check_if_field_is_list,
    detect_and_register_edges,
)
from src.backend.base.langflow.components.LangGraph.utils.memory_func import extract_memory, store_memory
from src.backend.base.langflow.components.LangGraph.utils.prompt_func import (
    form_memory_str_for_prompt,
    format_all_prompts,
)
from trustcall import create_extractor

from langflow.custom import Component
from langflow.io import (
    BoolInput,
    DropdownInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    Output,
)
from langflow.schema.dotdict import dotdict


class GraphNodeForAgent(Component):
    display_name = "Graph Node For Agent"
    description = "Node for LangGraph that processes input with agents component."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "LangChain"
    name = "LangGraphNodeAgent"

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
            name="agent_component",
            input_types=["AgentComponentForLangGraph"],
            display_name="Agent",
            info="Connect an Agent component to use its functionality.",
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
            "Use syntax {langflow_model_schema} to include the model's verbose schema in Agent Instruction and Agent Input",  # noqa: E501
            input_types=["ModelClassWrapper"],
        ),
        DropdownInput(
            name="output_type",
            display_name="Output Type",
            info="Select the output type for this node. If you select Pydantic, the output will be a Pydantic model. "
            "If you select State Field, you can assign the output (single value) "
            "to a state field by manually typing the field name.",
            options=["Pydantic", "State Field"],
            value="Pydantic",
            real_time_refresh=True,
        ),
        BoolInput(
            name="use_pydantic_extractor",
            display_name="Use Separate Pydantic Extractor?",
            info="If you are not using the built-in structured agent, you can enable this option to use a separate "
            "Pydantic extractor for better JSON parsing and validation. "
            "Use only with normal agent. If using structured agent, this option will be ignored.",
            value=False,
            show=True,
            dynamic=True,
            real_time_refresh=True,
        ),
        HandleInput(
            name="extraction_llm",
            display_name="Extraction Language Model",
            info="Language Model Used As Separate Extractor",
            input_types=["LanguageModel"],
            required=False,
            show=False,
            dynamic=True,
        ),
        MessageTextInput(
            name="output_state_field",
            display_name="Output State Field",
            info="If you choose 'State Field' as the output type, "
            "enter the field name where the output should be stored. "
            "If an output state model is defined for the node, the field name must match one of its fields. "
            "By default, the graph's output state will be used.",
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
            "pending tasks are completed. "
            "This is particularly relevant when branches have different lengths, which is common in "
            "workflows like map-reduce flows. "
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
        Output(display_name="Agent Node", name="next_node", method="build_graph")
    ]

    def _update_class_identity(self):
        """Update the class identity based on whether command addon is connected."""
        if hasattr(self, "return_command_addon") and self.return_command_addon:
            # Change the class name for detection by other components
            self.__class__.__name__ = "GraphNodeForAgentWithCommand"
            self.name = "LangGraphNodeAgentWithCommand"
            self.display_name = "Graph Node For Agent With Command"
        else:
            # Reset to original class name
            self.__class__.__name__ = "GraphNodeForAgent"
            self.name = "LangGraphNodeAgent"
            self.display_name = "Graph Node For Agent"


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


    async def run_agent(self, state, _original_prompts, runtime) -> dict[str, Any]:
        # Create local copies of prompts using the original templates passed from build_graph
        original_system_prompt, original_input_template = _original_prompts
        mem_str = form_memory_str_for_prompt(self.memories) if self.memories else ""

        # Format prompts locally (don't modify the shared component)
        formatted_system_prompt = format_all_prompts(original_system_prompt, mem_str, runtime, state, self.node_name)
        print("Debug formatted system prompt:", formatted_system_prompt)  # noqa: T201

        # ALWAYS use the original template, never the current agent component text\
        formatted_input_text = format_all_prompts(original_input_template, mem_str, runtime, state, self.node_name)
        print("Debug formatted input text:", formatted_input_text)  # noqa: T201

        # Set formatted prompts to the agent
        self.agent_component.system_prompt = formatted_system_prompt
        self.agent_component.input_value.text = formatted_input_text
        self.agent_component.input_value.sender = "User"

        # Run the agent using the shared instance with temporary values
        if self.output_type == "Pydantic":
            self.agent_component.output_model = self.output_model
            if self.agent_component._agent_type == "normal":
                if not self.use_pydantic_extractor:
                    msg = (
                        "Pydantic output is not supported for Normal Agent. "
                        "Change to State Field output type, use separate extractor or use Structured Agent."
                    )
                    raise ValueError(msg)
                response = await self.agent_component.message_response()
                response = self.extract_pydantic_with_separated_llm(
                    response.text, self.form_extraction_system_prompt(self.agent_component.format_instructions)
                    )
                response = await self.agent_component.build_structured_output_base(response)
            else:
                response = await self.agent_component.json_response()
        elif self.output_type == "State Field":
            if self.agent_component._agent_type == "structured":
                msg = (
                    "State Field output is not supported for Structured Agent. "
                    "Change to Pydantic output type or use Normal Agent."
                )
                raise ValueError(msg)
            response = await self.agent_component.message_response()

        # Rest of your processing code...
        if self.output_type == "Pydantic":
            return response
        is_list_field = check_if_field_is_list(self.output_state, self.output_state_field)
        if is_list_field:
            return {self.output_state_field: [response.text]}
        return {self.output_state_field: response.text}


    # Add this node to builder
    def build_graph(self) -> "GraphNodeForAgent":
        # Get the shared builder from context
        builder = self.graph_builder
        if not builder:
            msg = (
                f"{self.node_name} | No StateGraph builder found in context. "
                "Make sure CreateStateGraph component is connected and executed first."
            )
            raise ValueError(msg)

        command_type_hint = self.return_command_addon.type_hint if self.return_command_addon else None

        # Capture original prompts at build time (clean approach - no instance variables).
        # This is useful when using with Send API because original
        # prompt always preserved and not be modified in runtime.
        original_prompts = (
            self.agent_component.system_prompt,
            self.agent_component.input_value.text
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
            # Run the agent and get results
            try:
                # Try to get memories
                self.memories = await extract_memory(self.get_from_mem_addon, self.memories, store, config)
                # Get result
                result = await self.run_agent(state, original_prompts, runtime)
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
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                msg = f"{self.node_name} | Exception: {e}"
                raise ValueError(msg) from e
            return result


        params = build_params_for_add_node(self.node_caching, self.retry_policy, self.defer_node)
        builder.add_node(
            self.node_name, node_function, **params
        )
        print(f"Added node: {self.node_name}")  # noqa: T201

        # THEN detect and register edges (after node exists)
        self.graph_builder = detect_and_register_edges(builder, self.node_name, self.previous_nodes)
        return self


    def extract_pydantic_with_separated_llm(self, unstructured_text, extraction_system_prompt):
        if not hasattr(self.extraction_llm, "with_structured_output"):
            msg = (f"{self.node_name} | "
                   "Language model does not support structured output to be able to extract Pydantic result. "
                   "Try another Extraction LLM.")
            raise TypeError(msg)
        try:
            llm_with_structured_output = create_extractor(self.extraction_llm, tools=[self.output_model])
        except NotImplementedError as exc:
            msg = f"{self.node_name} | {self.extraction_llm.__class__.__name__} does not support structured output."
            raise TypeError(msg) from exc
        try:
            result = get_chat_result(
                runnable=llm_with_structured_output,
                system_message=extraction_system_prompt,
                input_value=unstructured_text
            )
            print("Debug extraction prompt:", extraction_system_prompt)  # noqa: T201
            # Handle the response
            if not isinstance(result, dict):
                msg = f"{self.node_name} | Invalid response format"
                raise TypeError(msg)
            # Extract response from result and get base model response
            responses = result.get("responses", [])
            if not responses:
                # Try to extract str result frome AIMessage if available
                messages = result.get("messages", [])
                if messages:
                    ai_message = messages[0]
                    content = getattr(ai_message, "content", None)
                if content:
                    # Return json in string format. _clarify_result will try to parse it
                    return content
                msg = "{self.node_name} | No responses found"
                raise ValueError(msg)
            return responses[0]
        except (ValueError, TypeError, KeyError, AttributeError, IndexError, RuntimeError, ImportError) as e:
            msg = f"{self.node_name} | Error in structured output extraction: {e}"
            raise ValueError(msg) from e


    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None):
        if field_name == "output_type":
            if field_value == "Pydantic":
                build_config["use_pydantic_extractor"]["show"] = True
                if build_config["use_pydantic_extractor"]["value"]:
                    build_config["extraction_llm"]["show"] = True
                    build_config["extraction_llm"]["required"] = True
                else:
                    build_config["extraction_llm"]["show"] = False
                    build_config["extraction_llm"]["required"] = False
                build_config["output_state_field"]["show"] = False
                build_config["output_state_field"]["required"] = False
            elif field_value == "State Field":
                build_config["use_pydantic_extractor"]["show"] = False
                build_config["extraction_llm"]["show"] = False
                build_config["extraction_llm"]["required"] = False
                build_config["output_state_field"]["show"] = True
                build_config["output_state_field"]["required"] = True

        if field_name == "use_pydantic_extractor":
            if field_value:
                build_config["extraction_llm"]["show"] = True
                build_config["extraction_llm"]["required"] = True
            else:
                build_config["extraction_llm"]["show"] = False
                build_config["extraction_llm"]["required"] = False

        if field_name == "use_mem":
            if field_value:
                build_config["get_from_mem_addon"]["show"] = True
                build_config["put_to_mem_addon"]["show"] = True
            else:
                build_config["get_from_mem_addon"]["show"] = False
                build_config["put_to_mem_addon"]["show"] = False

        return build_config
