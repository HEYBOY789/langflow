import re  # noqa: N999

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from lfx.custom.custom_component.component import Component
from lfx.io import DropdownInput, HandleInput, MessageTextInput, MultilineInput, NestedDictInput, Output
from src.backend.base.langflow.components.LangGraph.utils.memory_func import config_namespace


class StoreLongMemAddonForGraphNode(Component):
    display_name = "Store Long Memory Addon"
    description = "Stores long-term memory for a user. Postgres will be used as database, make sure to set up the Postgres database first."  # noqa: E501
    documentation: str = "https://langchain-ai.github.io/langgraph/concepts/memory/#long-term-memory"
    icon = "memory-stick"
    name = "StoreLongMemAddonForGraphNode"

    inputs = [
        MessageTextInput(
            name="name_space",
            display_name="Namespace",
            info="Memories are namespaced by a tuple. The namespace can be any length and represent anything, does not have to be user specific.\n"  # noqa: E501
            "Use syntax {name} to create dynamic namespaces. Read here: https://langchain-ai.github.io/langmem/guides/dynamically_configure_namespaces/#common-patterns",
            required=True,
            is_list=True
        ),
        DropdownInput(
            name="store_options",
            display_name="Store Options",
            info="Select the option to store memory.\n"
            "There are 2 options which are manual and using prebuilt langmem agent.\n"
            "Manual will directly store the output of the node as is.\n"
            "Prebuilt langmem agent will take that output, turn it into a message according to the way you set it up, and then store the message verbosely or as schema if you have set up the schema.",  # noqa: E501
            options=["Manual", "Prebuilt LangMem Agent"],
            value="Manual",
            real_time_refresh=True
        ),
        MessageTextInput(
            name="mem_id",
            display_name="Memory ID",
            info="The unique identifier for the memory entry.",
            required=True,
            show=True,
            dynamic=True
        ),
        NestedDictInput(
            name="extra_mem",
            display_name="Extra Memory Data",
            info="Additional data to store with the memory entry. Make sure the extra keys do not conflict with the keys in output state of the node.",  # noqa: E501
            show=True,
            dynamic=True
        ),
        MessageTextInput(
            name="field_embed",
            display_name="Specific Fields To Embed",
            info="When specified, only embed and store these fields into the memory vector database. "
            "Note: Normal database still embeds all fields. "
            "If left empty, all fields will be embedded and stored. "
            "Only work when using with embeding models. ",
            is_list=True,
            show=True,
            dynamic=True
        ),
        HandleInput(
            name="mem_manager_addon",
            display_name="Memory Store Manager Addon",
            info="Connect the Memory Store Manager Addon here if you are using Prebuilt LangMem Agent option.",
            input_types=["MemoryStoreManagerAddonForLangGraph"],
            dynamic=True,
            required=False,
            show=False
        ),
        MultilineInput(
            name="mem_mess_template",
            display_name="Custom Memory Message Template",
            info="Custom message template to format the memory before storing.\n"
            "Use {variable} to represent the field in the output of the node.\n"
            "Example: If your output has fields 'id' and 'content', you can use the format '{id}: {content}' to include both fields in the template.",  # noqa: E501
            required=False,
            dynamic=False,
            show=False,
        )
    ]

    outputs = [
        Output(display_name="Memories", name="output", method="build_output"),
    ]

    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "store_options":
            if field_value == "Manual":
                build_config["mem_id"]["required"] = True
                build_config["mem_id"]["show"] = True
                build_config["extra_mem"]["show"] = True
                build_config["field_embed"]["show"] = True
                build_config["mem_manager_addon"]["required"] = False
                build_config["mem_manager_addon"]["show"] = False
                build_config["mem_mess_template"]["show"] = False
                build_config["mem_mess_template"]["required"] = False
            elif field_value == "Prebuilt LangMem Agent":
                build_config["mem_id"]["required"] = False
                build_config["mem_id"]["show"] = False
                build_config["extra_mem"]["show"] = False
                build_config["field_embed"]["show"] = False
                build_config["mem_manager_addon"]["required"] = True
                build_config["mem_manager_addon"]["show"] = True
                build_config["mem_mess_template"]["show"] = True
                build_config["mem_mess_template"]["required"] = True
        return build_config

    def form_message_for_manager(self, memories_dict: dict):
        # Extract variable placeholders (now won't match escaped braces)
        placeholders = re.findall(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", self.mem_mess_template)

        for placeholder in placeholders:
            if placeholder not in memories_dict:
                msg = (
                        f"Placeholder '{{{placeholder}}}' not found in the output of the node. "
                        f"Check your message template for Memory Manager and ensure its variables match the fields in the output of the node.\n"  # noqa: E501
                        f"Available fields: [{', '.join(memories_dict.keys())}]"
                    )
                raise ValueError(msg)
        return self.mem_mess_template.format(**memories_dict)


    def build_output(self) -> "StoreLongMemAddonForGraphNode":
        # Pre-create memory manager if using LangMem Agent option
        # No need to config namespace because langmem will do it automatically
        if self.store_options == "Prebuilt LangMem Agent":
            from langmem import create_memory_store_manager
            manager_input_params = self.mem_manager_addon.manager_input_params
            manager_input_params["namespace"] = tuple(self.name_space)
            # Model is positional only parameter, so we need to pop it and pass it separately
            model = manager_input_params.pop("model")
            memory_manager = create_memory_store_manager(model, **manager_input_params)

        async def store_memory(store: BaseStore, memories_dict: dict, config: RunnableConfig | None=None):
            if self.store_options == "Manual":
                # Config Namespace
                ns_ = config_namespace(self.name_space, config)
                # Unpack extra_mem if provided
                if self.extra_mem:
                    memories_dict.update(self.extra_mem)
                # Structure memory for database consistency between manual and langmem agent methods
                structured_mem = {
                    "content": memories_dict
                }
                # Store memory in the specified namespace with the given ID
                await store.aput(namespace=ns_, key=self.mem_id, value=structured_mem, index=self.field_embed or None)
            elif self.store_options == "Prebuilt LangMem Agent":
                message = self.form_message_for_manager(memories_dict)
                await memory_manager.ainvoke({"messages": [message]})
        self.store_mem_func = store_memory
        return self
