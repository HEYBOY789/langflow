from langchain_core.runnables import RunnableConfig  # noqa: N999
from langgraph.store.base import BaseStore
from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, IntInput, MessageTextInput, MultilineInput, NestedDictInput, Output
from src.backend.base.langflow.components.LangGraph.utils.memory_func import config_namespace


class GetLongMemAddonForGraphNode(Component):
    display_name = "Get Long Memory Addon"
    description = "Retrieves long-term memory for a user. Postgres will be used as database, make sure to set up the Postgres database first."  # noqa: E501
    documentation: str = "https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory"
    icon = "LangChain"
    name = "GetLongMemAddonForGraphNode"

    inputs = [
        MessageTextInput(
            name="name_space",
            display_name="Namespace",
            info="Namespace of memory to get or search for.\n"
            "Use syntax {name} to create dynamic namespaces. Read here: https://langchain-ai.github.io/langmem/guides/dynamically_configure_namespaces/#common-patterns",
            required=True,
            is_list=True
        ),
        DropdownInput(
            name="operation",
            display_name="Search or Get",
            info="Choose whether to search for relevant memories or get a specific memory by ID.",
            options=["Get", "Search"],
            value="Get",
            real_time_refresh=True
        ),
        MessageTextInput(
            name="mem_id",
            display_name="Memory ID",
            info="ID of the memory to get or search for.",
            required=True,
            dynamic=True,
            show=True
        ),
        NestedDictInput(
            name="filter",
            display_name="Filter For The Search",
            info="Specify criteria to filter memories during the search.",
            dynamic=True,
            show=False
        ),
        MessageTextInput(
            name="query",
            display_name="Query For The Search",
            info="Query string to search for relevant memories.",
            dynamic=True,
            show=False,
            required=False
        ),
        IntInput(
            name="limit",
            display_name="Limit",
            info="Maximum number of memories to retrieve.",
            value=3,
            show=False
        ),
        MultilineInput(
            name="mem_format",
            display_name="Format of Retrieved Memory",
            info="Format the retrieved memories into a string to be included in the agent's prompt.\n"
            "The format will be applied to each memory item.\n"
            "Use {variable} to represent the field in the memory.\n"
            "Example: If your memory has fields 'id' and 'content', you can use the format '{id}: {content}' to include both fields in the prompt.\n"  # noqa: E501
            "Use syntax {langflow_mem_data} to include memories data into Agent Prompts, CrewAI Agent Prompts, CrewAI Tasks Prompts, User Prompts. " # noqa: E501
            "For Function, this will be passed as parameter.",
            value="{id}: {content}",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Memories", name="output", method="build_output"),
    ]


    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "operation":
            if field_value == "Get":
                build_config["mem_id"]["required"] = True
                build_config["mem_id"]["show"] = True
                build_config["filter"]["show"] = False
                build_config["query"]["required"] = False
                build_config["query"]["show"] = False
                build_config["limit"]["show"] = False
            elif field_value == "Search":
                build_config["mem_id"]["required"] = False
                build_config["mem_id"]["show"] = False
                build_config["filter"]["show"] = True
                build_config["query"]["required"] = True
                build_config["query"]["show"] = True
                build_config["limit"]["show"] = True
        return build_config


    def build_output(self) -> "GetLongMemAddonForGraphNode":
        async def get_memory(store: BaseStore, config: RunnableConfig | None=None):
            ns_ = config_namespace(self.name_space, config)
            if self.operation == "Get":
                return await store.aget(ns_, self.mem_id)
            if self.operation == "Search":
                return await store.asearch(ns_, query=self.query, filter=self.filter, limit=self.limit)
            return None
        self.get_mem_func = get_memory
        return self
