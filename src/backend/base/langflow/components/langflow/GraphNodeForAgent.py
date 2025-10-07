import re  # noqa: N999
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START
from langgraph.store.base import BaseStore
from lfx.base.models.chat_result import get_chat_result
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


    def _detect_and_register_edges(self):
        """Detect edges based on previous_nodes connections and add them directly to builder."""
        builder = self.graph_builder
        if not builder:
            msg = f"{self.node_name} | No StateGraph builder found in context."
            raise ValueError(msg)

        # Create edges from previous nodes to this node (if any)
        if self.previous_nodes:
            for prev_node_component in self.previous_nodes:
                # Create a start node if previous node is from CreateStateGraphComponent
                if prev_node_component.__class__.__name__ == "CreateStateGraphComponent":
                    builder.add_edge(START, self.node_name)
                    print(f"Added START edge: START -> {self.node_name}")  # noqa: T201
                    continue

                # If previous_node is conditonal edge, Send API or Node that return Command, just pass because all the"
                # logic is handled in ConditionalEdgeForLangGraph
                if prev_node_component.__class__.__name__ in [
                    "ConditionalEdgeForLangGraph",
                    "SendMapReduceForLangGraph",
                    "GraphNodeForAgentWithCommand",
                    "GraphNodeForCrewAIAgentWithCommand",
                    "GraphNodeForCrewAICrewWithCommand",
                    "GraphNodeForFunctionWithCommand",
                    "GraphNodeAsSubGraphWithCommand"
                ]:
                    print(f"Skipping {prev_node_component.__class__.__name__}")  # noqa: T201
                    continue

                # Get the node name from the previous GraphNode component
                prev_node_name = prev_node_component.node_name
                builder.add_edge(prev_node_name, self.node_name)
                print(f"Added edge: {prev_node_name} -> {self.node_name}")  # noqa: T201


    def format_prompt(self, prompt_template, state) -> str:
        """Format the prompt by replacing placeholders with actual values from the state object.

        Use {{}} to escape literal braces in schema, and {} for variables.
        """
        # First, temporarily replace escaped braces {{}} with a placeholder
        temp_placeholder = "___ESCAPED_BRACE___"
        formatted_prompt = prompt_template.replace(
            "{{", f"{temp_placeholder}OPEN").replace("}}", f"{temp_placeholder}CLOSE"
            )

        # Extract variable placeholders (now won't match escaped braces)
        placeholders = re.findall(r"{([a-zA-Z_][a-zA-Z0-9_.]*?)}", formatted_prompt)
        placeholders = list(set(placeholders))  # Remove duplicates

        from pydantic import BaseModel
        for placeholder in placeholders:
            # Check if the placeholder exists as an attribute in the state object
            if "." not in placeholder:
                if not hasattr(state, placeholder):
                    msg = (
                        f"{self.node_name} | Placeholder '{{{placeholder}}}' not found in the input state of node. "
                        f"Check your prompt template and ensure it matches the state attributes.\n"
                        f"Available attributes: [{', '.join(state.__dict__.keys())}]"
                    )
                    raise ValueError(msg)
                # Replace the placeholder with the actual value
                value = getattr(state, placeholder)
                formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", str(value))

            if "." in placeholder:
                var, field = placeholder.split(".")
                if not hasattr(state, var):
                    msg = (
                        f"{self.node_name} | Variable '{var}' not found in the input state of node. "
                        f"Check your prompt template and ensure it matches the state attributes.\n"
                        f"Available attributes: [{', '.join(state.__dict__.keys())}]"
                    )
                    raise ValueError(msg)


                if isinstance(getattr(state, var), dict):
                    if field not in getattr(state, var):
                        msg = (
                            f"{self.node_name} | Field '{field}' not found in the dictionary attribute '{var}'. "
                            f"Check your prompt template and ensure it matches the state attributes.\n"
                            f"Available fields in '{var}': [{', '.join(getattr(state, var).keys())}]"
                        )
                        raise ValueError(msg)
                    # Replace the placeholder with the actual value
                    value = getattr(state, var)[field]
                    formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", str(value))

                if isinstance(getattr(state, var), BaseModel):
                    if not hasattr(getattr(state, var), field):
                        msg = (
                            f"{self.node_name} | Field '{field}' not found in the model attribute '{var}'. "
                            f"Check your prompt template and ensure it matches the state attributes.\n"
                            f"Available fields in '{var}': [{', '.join(getattr(state, var).__dict__.keys())}]"
                        )
                        raise ValueError(msg)

                    # Replace the placeholder with the actual value
                    value = getattr(getattr(state, var), field)
                    formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", str(value))

        # Restore escaped braces
        return formatted_prompt.replace(f"{temp_placeholder}OPEN", "{").replace(f"{temp_placeholder}CLOSE", "}")



    def form_extraction_system_prompt(self, prompt_template):
        """Form the system prompt for structured output extraction. Use for extraction LLM when using AgentComponent."""
        verbose_str = self.output_state.schema.get("verbose_schema_str")

        # Replace {model_schema} with the actual verbose_str in system prompt (combined prompt if using pydantic)
        return prompt_template.replace("{langflow_model_schema}", verbose_str.strip())


    def _check_if_field_is_list(self, field_name: str) -> bool:
        """Check if a field is a list type using the schema."""
        schema_fields = self.output_state.schema.get("fields", [])

        for field in schema_fields:
            if field["name"] == field_name:
                field_type_str = field["type"]
                # Check if the type string contains "List"
                return "List[" in field_type_str or field_type_str == "list"
        return False


    async def extract_memory(self, store: BaseStore | None = None, config: RunnableConfig | None=None):
        # Format system prompt and input value with memory
        if store and self.get_from_mem_addon:
            for get_from_mem in self.get_from_mem_addon:
                mem = await get_from_mem.get_mem_func(store, config=config)
                mem_format = get_from_mem.mem_format
                if mem:
                    if isinstance(mem, list):
                        for m in mem:
                            if m not in self.memories:
                                self.memories.append({m: mem_format})
                    elif mem not in self.memories:
                        self.memories.append({mem: mem_format})
            # print(f"Retrieved memories: {self.memories}")

    def form_memory_str_for_prompt(self):
        mem_ = set()
        # print("len of memories:", len(self.memories))
        for mem in self.memories:
            for m, m_format in mem.items():
                # print("Initial memory content:", m)
                # Extract variable placeholders (now won't match escaped braces)
                placeholders = re.findall(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", m_format)

                m_val = getattr(m, "value", None)
                if not m_val:
                    continue

                # Get content in mem_val
                m_val_content = m_val.get("content", None)
                # print(f"Raw memory content: {mem_val_content}")
                if isinstance(m_val_content, dict):
                    for placeholder in placeholders:
                        if placeholder not in m_val_content:
                            print(f"Missing placeholder {{{placeholder}}} in memory. Using raw content.")  # noqa: T201
                            m_val_content = str(m_val_content)
                            # print(f"Formatted memory content as dict: {m_val_content}")
                            break
                    if isinstance(m_val_content, dict):
                        # If still dict, format using all keys
                        m_val_content = m_format.format(**m_val_content)
                elif isinstance(m_val_content, str):
                    # print(f"Formatted memory content as str: {m_val_content}")
                    m_val_content = str(m_val_content)

                mem_.add(m_val_content.strip())
        return "\n".join(mem_)

    async def store_memory(self, result: dict, store: BaseStore | None = None, config: RunnableConfig | None=None):
        if store and self.put_to_mem_addon:
            for put_mem_addon in self.put_to_mem_addon:
                await put_mem_addon.store_mem_func(store, result, config)

    def format_runtime_prompt(self, prompt, runtime):
        if not runtime.context:
            return prompt

        placeholders = re.findall(r"\{langflow_runtime_context\.([a-zA-Z_][a-zA-Z0-9_]*)\}", prompt)
        placeholders = list(set(placeholders))
        for placeholder in placeholders:
            if placeholder in runtime.context.__dict__:
                value = getattr(runtime.context, placeholder)
                # Convert to string, allowing for None, False, 0, empty strings
                prompt = prompt.replace(f"{{langflow_runtime_context.{placeholder}}}", str(value))
            else:
                msg = (
                    f"Runtime context field '{placeholder}' not found in context. "
                    f"Available fields: {list(runtime.context.__dict__.keys())}"
                )
                raise ValueError(msg)
        return prompt

    def format_all_prompts(self, prompt, mem_str, runtime, state):
        formatted_prompt = self.form_extraction_system_prompt(prompt)
        formatted_prompt = formatted_prompt.replace("{langflow_mem_data}", mem_str)
        formatted_prompt = self.format_runtime_prompt(formatted_prompt, runtime)
        return self.format_prompt(formatted_prompt, state)

    async def run_agent(self, state, _original_prompts, runtime) -> dict[str, Any]:
        # Create local copies of prompts using the original templates passed from build_graph
        original_system_prompt, original_input_template = _original_prompts
        mem_str = self.form_memory_str_for_prompt() if self.memories else ""

        # Format prompts locally (don't modify the shared component)
        formatted_system_prompt = self.format_all_prompts(original_system_prompt, mem_str, runtime, state)
        print("Debug formatted system prompt:", formatted_system_prompt)  # noqa: T201

        # ALWAYS use the original template, never the current agent component text\
        formatted_input_text = self.format_all_prompts(original_input_template, mem_str, runtime, state)
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
        is_list_field = self._check_if_field_is_list(self.output_state_field)
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
                await self.extract_memory(store, config)
                # Get result
                result = await self.run_agent(state, original_prompts, runtime)
                # Store result as long memory
                await self.store_memory(result, store, config)
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

        # Add this node to the builder FIRST
        if self.node_caching > 0:
            from langgraph.types import CachePolicy
            caching_ = CachePolicy(ttl=self.node_caching)
        else:
            caching_ = None

        # Add retry policy if provided
        policy_ = self.retry_policy or None

        builder.add_node(
            self.node_name, node_function, cache_policy=caching_, retry_policy=policy_, defer=self.defer_node
            )
        print(f"Added node: {self.node_name}")  # noqa: T201

        # THEN detect and register edges (after node exists)
        self._detect_and_register_edges()
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
