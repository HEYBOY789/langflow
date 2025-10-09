from langmem import create_search_memory_tool  # noqa: N999
from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.inputs.inputs import MessageTextInput, MultilineInput
from lfx.io import Output


class MemorySearchToolComponent(LCToolComponent):
    display_name = "Memory Search Tool"
    description = "Create a tool for searching memories stored in a LangGraph BaseStore.\n"
    "When used with an agent, make sure the agent has been specifically tasked with retrieving memories."
    icon = "memory-stick"
    documentation = "https://langchain-ai.github.io/langmem/reference/tools/#langmem.create_search_memory_tool"
    name = "MemorySearchTool"

    inputs = [
        MessageTextInput(
            name="name_space",
            display_name="Namespace",
            info="Memories are namespaced by a tuple. The namespace can be any length and represent anything, does not have to be user specific.\n"  # noqa: E501
            "Use syntax {name} to create dynamic namespaces. Read here: https://langchain-ai.github.io/langmem/guides/dynamically_configure_namespaces/#common-patterns",
            required=True,
            is_list=True
        ),
        MultilineInput(
            name="instructions",
            display_name="Custom Instruction To Retrieve Memories",
            info="Custom instruction to retrieve memories from conversations. If not provided, a default instruction will be used.",  # noqa: E501
            value="Use this tool to retrieve relevant memories about American's history when asked."
        ),
    ]

    outputs = [
        Output(display_name="Tool", name="result_tool", method="build_tool"),
    ]

    def build_tool(self) -> Tool:
        config = {}
        ns = tuple(self.name_space)
        config["namespace"] = ns

        if self.instructions:
            config["instructions"] = self.instructions

        return create_search_memory_tool(**config)
