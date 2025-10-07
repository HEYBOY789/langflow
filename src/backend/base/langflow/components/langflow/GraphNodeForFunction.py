import re  # noqa: N999
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START
from langgraph.store.base import BaseStore

from langflow.custom import Component
from langflow.io import BoolInput, HandleInput, IntInput, MessageTextInput, Output


class GraphNodeForFunction(Component):
    display_name = "Graph Node For Function"
    description = "Node for LangGraph that processes input with python functions component."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "LangChain"
    name = "LangGraphNodeFunction"

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
            name="function_",
            input_types=["Callable"],
            display_name="Function",
            info="Connect a Python function to use its logic in Langflow.\n"
            "Langflow's Python Function component supports only a single function.\n"
            "Always wrap your code inside an async function to ensure compatibility and make sure to return from it.\n"
            "Use await for any async operations inside the wrapper function instead of asyncio.run.\n"
            "Must return a dict with keys matching the output state model fields.\n"
            "Must have (state, output_types, memories, model_schema, runtime) as parameters for the async wrapper function. Order matter.\n"  # noqa: E501
            "* state (BaseModel): The input state of node.\n"
            "* output_types (dict): A dict mapping output field names to their types. "
            "Example: {'person': Person, number: int}. "
            "Useful when you want to create instances of types in output state.\n"
            "* memories (str): Optional memory format as a string for the node.\n"
            "* model_schema (str): Optional model schema format as a string for the node.\n"
            "* runtime (Runtime): Runtime Config. "
            "Use 'runtime.context.field_name' to access value of field in runtime config. "
            "If no runtime config, runtime.context will be None.\n"
            "Example: \n"
            "Input state has 'second' field\n"
            "Output state has 'result' field\n"
            "```code```\n"
            "async def f_(state, output_types, memories, model_schema, runtime):\n"
            "____async def stop_for_second(state):\n"
            "________await asyncio.sleep(state.second)\n"
            '________return f"Stop for {state.second} seconds"\n'
            "____result = await stop_for_second(state)\n"
            '____return {"result": result}\n'
            "```code```\n"
            "```code```\n"
            "async def conv_schema_to_instance_for_overallstate(state, output_types, memories, model_schema, runtime):"
            '____person = output_types["person"](**state.__dict__)\n'
            '____return {"person": [person]}\n'
            "```code```\n",
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
            "This is particularly relevant when branches have different lengths, which is common in workflows "
            "like map-reduce flows. "
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
        Output(display_name="Function Node", name="next_node", method="build_graph")
    ]

    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "use_mem":
            if field_value:
                build_config["get_from_mem_addon"]["show"] = True
                build_config["put_to_mem_addon"]["show"] = True
            else:
                build_config["get_from_mem_addon"]["show"] = False
                build_config["put_to_mem_addon"]["show"] = False
        return build_config

    def _update_class_identity(self):
        """Update the class identity based on whether command addon is connected."""
        if hasattr(self, "return_command_addon") and self.return_command_addon:
            # Change the class name for detection by other components
            self.__class__.__name__ = "GraphNodeForFunctionWithCommand"
            self.name = "LangGraphNodeFunctionWithCommand"
            self.display_name = "Graph Node For Function With Command"
        else:
            # Reset to original class name
            self.__class__.__name__ = "GraphNodeForFunction"
            self.name = "LangGraphNodeFunction"
            self.display_name = "Graph Node For Function"


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
            # If previous_node is conditonal edge,
            # just pass because all the logic is handled in ConditionalEdgeForLangGraph
            for prev_node_component in self.previous_nodes:
                # Create a start node if previous node is from CreateStateGraphComponent
                if prev_node_component.__class__.__name__ == "CreateStateGraphComponent":
                    builder.add_edge(START, self.node_name)
                    print(f"Added START edge: START -> {self.node_name}")  # noqa: T201
                    continue

                # If previous_node is conditonal edge,
                # Send API or Node that return Command, just pass because all the logic is handled in
                # ConditionalEdgeForLangGraph
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

    # Add this node to builder
    def build_graph(self) -> "GraphNodeForFunction":
        # Get the shared builder from context
        builder = self.graph_builder
        if not builder:
            msg = ("{self.node_name} | No StateGraph builder found in context. "
                   "Make sure CreateStateGraph component is connected and executed first.")
            raise ValueError(msg)

        command_type_hint = self.return_command_addon.type_hint if self.return_command_addon else None

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
            # Run the function and get result
            try:
                # Try to get memories
                await self.extract_memory(store, config)
                # Get result
                verbose_str = self.output_state.schema.get("verbose_schema_str")
                mem_str = self.form_memory_str_for_prompt() if self.memories else ""
                result = await self.function_(state, self.output_state.get_field_types(), mem_str, verbose_str, runtime)
                # Store result as long memory
                await self.store_memory(result, store, config)
                # Return Command object if command_addon is provided
                if self.return_command_addon:
                    # Convert result to object to synchonize when using .field
                    if isinstance(result, dict):
                        result = self.output_model(**result)
                    result = self.return_command_addon.function_(result, state)
                else:
                    pass
            except Exception as e:
                error_msg = f"{self.node_name} | Exception: {e}"
                raise ValueError(error_msg) from e
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




