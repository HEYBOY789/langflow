from langmem import create_manage_memory_tool  # noqa: N999
from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.inputs.inputs import BoolInput, HandleInput, MessageTextInput
from lfx.io import Output


class MemoryManagerToolComponent(LCToolComponent):
    display_name = "Memory Manager Tool"
    description = "Create a tool for managing persistent memories in conversations.\n"
    "When used with an agent, make sure the agent have to specifically tasked in storing memories."
    icon = "memory-stick"
    documentation = "https://langchain-ai.github.io/langmem/reference/tools/#langmem.create_manage_memory_tool"
    name = "MemoryManagerTool"

    inputs = [
        MessageTextInput(
            name="name_space",
            display_name="Namespace",
            info="Memories are namespaced by a tuple. The namespace can be any length and represent anything, does not have to be user specific.\n"  # noqa: E501
            "Use syntax {name} to create dynamic namespaces. Read here: https://langchain-ai.github.io/langmem/guides/dynamically_configure_namespaces/#common-patterns",
            required=True,
            is_list=True
        ),
        MessageTextInput(
            name="instructions",
            display_name="Custom Instruction To Extract Memories",
            info="Custom instruction to extract memories from conversations. If not provided, a default instruction will be used.",  # noqa: E501
            value="Use this tool to store relevant memories about American's history."
        ),
        HandleInput(
            name="schema",
            display_name="Schema",
            info="A Pydantic model defining the structure of memory entries. "
            "Remember to include the docstring as guidance in your models or LLM can not use them.",
            input_types=["ModelClassWrapper"],
        ),
        BoolInput(
            name="create",
            display_name="Enable Create",
            info="Whether to allow creating new memory entries. When False, the manager will only update existing memories.",  # noqa: E501
            value=True
        ),
        BoolInput(
            name="update",
            display_name="Enable Update",
            info="Whether to allow creating new memory entries. When False, the manager will only update existing memories.",  # noqa: E501
            value=True
        ),
        BoolInput(
            name="delete",
            display_name="Enable Delete",
            info="Whether to allow creating new memory entries. When False, the manager will only update existing memories.",  # noqa: E501
            value=True
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

        if self.schema:
            config["schema"] = self.schema.model_class

        actions_permitted = []
        if self.create:
            actions_permitted.append("create")
        if self.update:
            actions_permitted.append("update")
        if self.delete:
            actions_permitted.append("delete")
        if actions_permitted:
            config["actions_permitted"] = tuple(actions_permitted)

        return create_manage_memory_tool(**config)
