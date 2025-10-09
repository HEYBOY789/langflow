from typing import Union  # noqa: N999

from langgraph.graph import END
from langgraph.store.postgres.aio import AsyncPostgresStore
from lfx.schema.data import Data
from src.backend.base.langflow.components.LangGraph.utils.graph_node_func import detect_and_register_edges
from src.backend.base.langflow.components.LangGraph.utils.input_convert_func import normalize_input_data

from langflow.custom import Component
from langflow.io import BoolInput, DictInput, DropdownInput, HandleInput, IntInput, MessageTextInput, Output


class AsyncStoreManager:
    _instance = None
    _store_context = None
    _store = None
    _index = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def setup_store(self, db_url, index: dict | None = None):
        if self._store_context is None:
            # Store index configuration
            self._index = index

            # Get the async context manager
            self._store_context = AsyncPostgresStore.from_conn_string(db_url, index=index)

            # Manually call __aenter__() for async context manager
            self._store = await self._store_context.__aenter__()

            # Initialize the store
            await self._store.setup()

        return self._store

    async def cleanup_store(self):
        if self._store_context:
            # Manually call __aexit__() for async context manager
            await self._store_context.__aexit__(None, None, None)
            self._store_context = None
            self._store = None


class GraphRun(Component):
    display_name = "Graph Runner"
    description = "Running the graph"
    icon = "LangChain"
    name = "GraphRunner"

    inputs = [
        HandleInput(
            name="final_nodes",
            display_name="Final Nodes",
            info="Connect the final GraphNode(s) that should link to END. Can connect multiple nodes.",
            input_types=[
                "GraphNodeAsSubGraph",
                "GraphNodeForCrewAIAgent",
                "GraphNodeForAgent",
                "GraphNodeForCrewAICrew",
                "GraphNodeForFunction",
                "ConditionalEdgeForLangGraph",
                "AddEdgeToLoopNodeForLangGraph",
                "GraphNodeForAgentWithCommand",
                "GraphNodeForCrewAIAgentWithCommand",
                "GraphNodeForCrewAICrewWithCommand",
                "GraphNodeForFunctionWithCommand",
                "GraphNodeAsSubGraphWithCommand"
                ],
            is_list=True,
            required=True
        ),
        HandleInput(
            name="custom_structure_data",
            display_name="Custom Class",
            info="Connect a custom classes (via structure data component) to use their instance when create input data",
            input_types=["ModelClassWrapper"],
            is_list=True,
            advanced=True,
        ),
        DropdownInput(
            name="input_type",
            display_name="Input Type",
            info="Select Data or Dictionary as input type for the graph. \n"
            "Dictionary should be used for quick invoke.\n"
            "Data should be used to synchronize datatype with other flow's components. "
            "Example: When you want to run a loop.",
            options=["Dictionary", "Data Type"],
            value="Dictionary",
            real_time_refresh=True
        ),
        DictInput(
            name="input_dict",
            display_name="Input Data",
            info="Specify the input data for the graph or subgraph. "
            "If subgraph have the same state schemas as the parent graph, this field will not be used because the input"
            " data will be taken from the parent graph state automatically.\n\n"
            "• Use @is_list to mark a list.\n"
            "Example: @is_list [1, 2, 3]\n\n"
            "• Use @is_dict to mark a dictionary.\n"
            'Example: @is_dict {"name": "John", "age": 30}\n\n'
            "• Use @is_class to mark a custom class type.\n"
            'Example: @is_class Company{"name": "TechCorp", "employees": 100, "public": false}\n\n'
            "• Use @is_bool to mark a boolean value.\n"
            "Example: @is_bool True\n\n"
            "• Use @is_parent_state to access the value of the field in parent graph state.\n"
            "Example: @is_parent_state field_name\n\n"
            "• For nested structures (like a list or dict of custom classes or involve in using a value of parent"
            " graph state), define each class instance or retrieve value from parent state first, assign it to a "
            "variable using as {variable}, and then reference those variables inside your list, dict, "
            "or another class. Separate multiple declarations using ||.\n"
            'Example: @is_parent_state person as p1 || @is_class Person{"name": "a", "age": 8} as p2 || '
            '@is_class Person{"name": "b", "age": 9} as p3 || @is_list [p1, p2, p3]\n'
            '@is_class Birthday{"date": 2, "month": 12, "year": 1994} as birthday || @is_class Person{"name": "Lan",'
            ' "age": 20, "birthday": birthday} as person || @is_list [person]\n\n'
            "• For simple types like strings, integers, and floats, just input the value directly\n."
            'Examples: "Hello", 1, 1.0\n\n'
            "Note: Use double quotes for strings and dict keys",
            is_list=True,
            value={"example": '@is_class Birthday{"date": 2, "month": 12, "year": 1994} as birthday || "'
            '"@is_class Person{"name": "Lan", "age": 20, "birthday": birthday} as person || @is_list [person]'},
            dynamic=True,
            show=True,
        ),
        HandleInput(
            name="input_data_type",
            display_name="Input Data",
            input_types=["Data"],
            info="Specify the input data for the graph or subgraph. "
            "If subgraph have the same state schemas as the parent graph, this field will not be used because the input"
            " data will be taken from the parent graph state automatically.\n\n"
            "• Use @is_list to mark a list.\n"
            "Example: @is_list [1, 2, 3]\n\n"
            "• Use @is_dict to mark a dictionary.\n"
            'Example: @is_dict {"name": "John", "age": 30}\n\n'
            "• Use @is_class to mark a custom class type.\n"
            'Example: @is_class Company{"name": "TechCorp", "employees": 100, "public": false}\n\n'
            "• Use @is_bool to mark a boolean value.\n"
            "Example: @is_bool True\n\n"
            "• Use @is_parent_state to access the value of the field in parent graph state.\n"
            "Example: @is_parent_state field_name\n\n"
            "• For nested structures (like a list or dict of custom classes or involve in using a value of parent"
            " graph state), define each class instance or retrieve value from parent state first, assign it to a "
            "variable using as {variable}, and then reference those variables inside your list, dict, "
            "or another class. Separate multiple declarations using ||.\n"
            'Example: @is_parent_state person as p1 || @is_class Person{"name": "a", "age": 8} as p2 || '
            '@is_class Person{"name": "b", "age": 9} as p3 || @is_list [p1, p2, p3]\n'
            '@is_class Birthday{"date": 2, "month": 12, "year": 1994} as birthday || @is_class Person{"name": "Lan",'
            ' "age": 20, "birthday": birthday} as person || @is_list [person]\n\n'
            "• For simple types like strings, integers, and floats, just input the value directly\n."
            'Examples: "Hello", 1, 1.0\n\n'
            "Note: Use double quotes for strings and dict keys",
            dynamic=True,
            show=False,
        ),
        DropdownInput(
            name="input_type_runtime",
            display_name="Input Type for Runtime Config",
            info="Select the type of input value.",
            options=["Dictionary", "Data Type"],
            real_time_refresh=True,
            value="Dictionary",
        ),
        DictInput(
            name="input_runtime_dict",
            display_name="Input Runtime",
            info="Input value for the runtime config.\n"
            "Use syntax {langflow_runtime_context.field_name} to include value of field in runtime config:\n"
            "- In Agent Prompts, User Prompts (when using with GraphNodeForAgent).\n"
            "- In Agent Prompts, Tasks and User Prompts (when using with GraphNodeForCrewAIAgent and "
            "GraphNodeForCrewAiCrew).\n"
            "For GraphNodeForFunction, read it own Function guide for more details.\n",
            show=True,
            dynamic=True,
            is_list=True
        ),
        HandleInput(
            name="input_runtime_data_type",
            display_name="Input Runtime",
            info="Input value for the runtime config.\n"
            "Use syntax {langflow_runtime_context.field_name} to include value of field in runtime config:\n"
            "- In Agent Prompts, User Prompts (when using with GraphNodeForAgent).\n"
            "- In Agent Prompts, Tasks and User Prompts (when using with GraphNodeForCrewAIAgent and "
            "GraphNodeForCrewAiCrew).\n"
            "For GraphNodeForFunction, read it own Function guide for more details.\n",
            input_types=["Data"],
            show=False,
            dynamic=True,
        ),
        BoolInput(
            name="node_caching",
            display_name="Using Node Caching?",
            info="Enable Caching For The Graph. (https://langchain-ai.github.io/langgraph/concepts/low_level/#node-caching).",
            advanced=True,
            value=False
        ),
        IntInput(
            name="recursion_limit",
            display_name="Recursion Limit",
            info="Set the recursion limit for the graph. 0 is no limit.",
            value=0,
            advanced=True,
        ),
        BoolInput(
            name="using_long_mem",
            display_name="Using Long Memory?",
            info="Enable if you want to use long memory inside the graph. Postgres will be used as database, make sure to set up the Postgres database first.",  # noqa: E501
            advanced=True,
            value=False,
            real_time_refresh=True
        ),
        HandleInput(
            name="embeddings",
            display_name="Using Embeddings For Long Memory?",
            info="Connect embeddings model to use for long-term memory. If you are not using embeddings, memories are still retrievable, but not searchable",  # noqa: E501
            input_types=["Embeddings"],
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="table_emb_dim",
            display_name="Embedding Dimension Of Table",
            info="The dimension of the table to store embeddings in PostGres. "
            "Make sure it is the same dimension with output dimension of embedding model. "
            "Default to 1536. ",
            value=1536,
            show=False,
            dynamic=True,
        ),
        MessageTextInput(
            name="postgres_db_url",
            display_name="Postgres Database URL",
            info="Enter the Postgres database URL.",
            placeholder="postgres://username:password@localhost:5432/dbname",
            value="postgresql://langflow_user:1234@localhost:5432/langflow_mem?sslmode=disable",
            show=False,
            required=False,
            dynamic=True
        ),
        DictInput(
            name="configurable_options",
            display_name="Configurable Options",
            info="Additional configuration for LangGraph.",
            advanced=True,
        )
    ]

    outputs = [
        Output(display_name="Graph", name="graph", method="build_graph"),
        Output(display_name="Result", name="result", method="get_result")
    ]


    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "using_long_mem":
            if field_value:
                build_config["table_emb_dim"]["show"] = True
                build_config["embeddings"]["show"] = True
                build_config["postgres_db_url"]["show"] = True
                build_config["postgres_db_url"]["required"] = True
            else:
                build_config["table_emb_dim"]["show"] = False
                build_config["embeddings"]["show"] = False
                build_config["postgres_db_url"]["show"] = False
                build_config["postgres_db_url"]["required"] = False

        if field_name == "input_type":
            if field_value == "Dictionary":
                build_config["input_dict"]["show"] = True
                build_config["input_data_type"]["show"] = False
            if field_value == "Data Type":
                build_config["input_dict"]["show"] = False
                build_config["input_data_type"]["show"] = True

        if field_name == "input_type_runtime":
            if field_value == "Dictionary":
                build_config["input_runtime_dict"]["show"] = True
                build_config["input_runtime_data_type"]["show"] = False
            else:
                build_config["input_runtime_dict"]["show"] = False
                build_config["input_runtime_data_type"]["show"] = True
        return build_config


    def _pre_run_setup(self):

        # Convert input_dict or input_data_type to input_data
        if self.input_type == "Dictionary":
            self.input_data = self.input_dict
        if self.input_type == "Data Type":
            self.input_data = self.input_data_type.data

        # Create a placeholder to store parent state if we connect this component to GraphNodeAsSubGraph and pass
        # in parent state from GraphNodeAsSubGraph
        self.parent_state = None

        self.graph_input_state = self.final_nodes[0].graph_input_state
        self.graph_output_state = self.final_nodes[0].graph_output_state
        self.graph_builder = self.final_nodes[0].graph_builder
        self.graph_context_schema = self.final_nodes[0].graph_context_schema

        # Convert input_runtime_dict or input_runtime_data_type to input_runtime
        if self.graph_context_schema:
            if self.input_type_runtime == "Dictionary":
                self.input_runtime = self.input_runtime_dict
            if self.input_type_runtime == "Data Type":
                self.input_runtime = self.input_runtime_data_type.data
        else:
            self.input_runtime = None

        self.config = {}
        # Setup config for LangGraph
        if self.configurable_options:
            self.config.update({"configurable": self.configurable_options})

        # Config recursion limit
        if self.recursion_limit > 0:
            self.config.update({"recursion_limit": self.recursion_limit})

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

        # build_index_for_long_mem
        if self.using_long_mem and self.embeddings:
            if not self.table_emb_dim:
                msg = "Please set the dimmension of the embedding table"
                raise ValueError(msg)

            self.mem_store_index = {
                "dims": self.table_emb_dim or 1536,
                "embed": self.embeddings,
            }
        else:
            self.mem_store_index = None


    async def build_graph(self) -> "GraphRun":
        """Build the complete StateGraph with all nodes and edges."""
        builder = self.graph_builder
        if not builder:
            msg = "No StateGraph builder found in context."
            raise ValueError(msg)

        detect_and_register_edges(builder, END, self.final_nodes)

        if self.node_caching:
            from langgraph.cache.memory import InMemoryCache
            caching_ = InMemoryCache()
        else:
            caching_ = None

        if self.using_long_mem:
            # Use async store manager
            store_manager = AsyncStoreManager.get_instance()
            store = await store_manager.setup_store(self.postgres_db_url, self.mem_store_index)
            graph = builder.compile(store=store, cache=caching_)
        else:
            graph = builder.compile(cache=caching_)

        self._graph = graph
        return self

    async def get_result(self) -> Data:
        """Run the graph with test data if requested."""
        try:
            if not hasattr(self, "_graph"):
                await self.build_graph()

            return await self._execute_graph_async()
        finally:
            # Cleanup store when main graph execution is complete
            if self.using_long_mem:
                await AsyncStoreManager.get_instance().cleanup_store()

    async def _execute_graph_async(self) -> Data:
        """Helper method to execute the graph asynchronously."""
        try:
            self.serialize_input()
            self.input_data = normalize_input_data(self.input_data, parent_state=self.parent_state)
            result = await self._graph.ainvoke(
                input=self.input_data,
                config=self.config,
                context=self.input_runtime
            )
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            msg = f"Check the input state: {e}"
            raise ValueError(msg) from e
        else:
            if result is None:
                data = {"success": True, "result": None, "message": "Output state is not set."}
                return Data(**data)

            return Data(success=True, result=result)

    async def _execute_graph_async_subgraph(self):
        try:
            self.serialize_input()
            self.input_data = normalize_input_data(self.input_data, parent_state=self.parent_state)
            result = await self._graph.ainvoke(
                input=self.input_data,
                config=self.config,
                context=self.input_runtime
            )
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            msg = f"Check the input state: {e}"
            raise ValueError(msg) from e

        return result


    def serialize_input(self):
        # Get the input model from shared context
        input_state = self.graph_input_state
        input_model = input_state.model_class

        if hasattr(input_model, "model_fields"):
            missing_fields = []
            for field_name, field in input_model.model_fields.items():
                # Check if field type is not Optional (doesn't have Union[X, None])
                field_type = field.annotation
                is_optional = (
                    hasattr(field_type, "__origin__") and
                    field_type.__origin__ is Union and
                    type(None) in field_type.__args__
                    )

                if not is_optional and field_name not in self.input_data:
                    missing_fields.append(field_name)

            if missing_fields:
                return {
                    "success": False,
                    "error": (f"Missing required fields in input data: [{', '.join(missing_fields)}]. "
                             f"Make sure the input data matches the input state of the Graph.")
                }
        return None

