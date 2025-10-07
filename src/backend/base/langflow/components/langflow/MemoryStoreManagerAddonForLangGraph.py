from lfx.custom.custom_component.component import Component  # noqa: N999
from lfx.io import (
    BoolInput,
    HandleInput,
    IntInput,
    MessageTextInput,
    NestedDictInput,
    Output,
)


class MemoryStoreManagerAddonForLangGraph(Component):
    display_name = "Memory Store Manager Addon"
    description = "Create a Manager Store for LangGraph. This component will be used with Store Long Memory Addon component."  # noqa: E501
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "memory-stick"
    name = "MemoryStoreManagerAddonForLangGraph"

    inputs = [
        HandleInput(
            name="llm",
            display_name="Language Model",
            info="Language model that will run the Manager.",
            input_types=["LanguageModel"],
            required=True,
        ),
        MessageTextInput(
            name="instruction",
            display_name="Custom Instruction To Extract Memories",
            info="Custom instruction to extract memories from conversations. If not provided, a default instruction will be used.",  # noqa: E501
            value=""
        ),
        BoolInput(
            name="use_schema",
            display_name="Using Schema",
            info="Turn on if you want to use schemas to structure memory entries",
            value=False,
            real_time_refresh=True
        ),
        HandleInput(
            name="schemas",
            display_name="Schemas",
            info="List of Pydantic models defining the structure of memory entries."
            "Remember to include the docstring as guidance in your models or LLM can not use them.",
            input_types=["ModelClassWrapper"],
            is_list=True,
            dynamic=True,
            required=False,
            show=False
        ),
        MessageTextInput(
            name="default_string",
            display_name="Default Memory",
            info="Default value in string to persist to the store if no other memories are found.",
            show=True,
            dynamic=True,
        ),
        NestedDictInput(
            name="default_dict",
            display_name="Default Memory",
            info="Default value in dict to persist to the store if no other memories are found. Should match one of the provided schemas.",  # noqa: E501
            dynamic=True,
            show=False,
            required=False
        ),
        HandleInput(
            name="default_factory",
            display_name="Default Memory Function",
            info="A factory function to generate the default value. "
            "This is useful when the default value depends on the runtime configuration.\n"
            "If not using schema, the function should return a string.\n"
            "If using schema, the function should return a dict that matches one of the provided schemas.\n"
            "Example:\n"
            "def get_configurable_default(config):\n"
            "____default_preference = config['configurable'].get('preference', "
            "'Use a concise and professional tone in all responses.')\n"
            "____return default_preference",
            input_types=["Callable"],
        ),
        BoolInput(
            name="insert",
            display_name="Enable Insert",
            info="Whether to allow creating new memory entries. When False, the manager will only update existing memories.",  # noqa: E501
            value=True
        ),
        BoolInput(
            name="delete",
            display_name="Enable Delete",
            info="Whether to allow deleting existing memories that are outdated or contradicted by new information.",
            value=True
        ),
        HandleInput(
            name="query_model",
            display_name="Query Model",
            info="Optional separate model for memory search queries. "
            "Using a smaller, faster model here can improve performance. "
            "If None, uses the primary model.",
            input_types=["LanguageModel"],
        ),
        IntInput(
            name="query_limit",
            display_name="Query Limit",
            info="Maximum number of relevant memories to retrieve for each conversation. "
            "Higher limits provide more context but may slow down processing.",
            value=5
        ),
    ]

    outputs = [
        Output(display_name="Memory Store Manager", name="output", method="build_output"),
    ]

    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "use_schema":
            if field_value:
                build_config["schemas"]["show"] = True
                build_config["schemas"]["required"] = True
                build_config["default_string"]["show"] = False
                build_config["default_dict"]["show"] = True
            else:
                build_config["schemas"]["show"] = False
                build_config["schemas"]["required"] = False
                build_config["default_string"]["show"] = True
                build_config["default_dict"]["show"] = False
        return build_config

    def build_output(self) -> "MemoryStoreManagerAddonForLangGraph":
        self.manager_input_params = {}
        self.manager_input_params["model"] = self.llm
        if self.instruction:
            self.manager_input_params["instructions"] = self.instruction

        if self.use_schema:
            schemas = [schema.model_class for schema in self.schemas]
            self.manager_input_params["schemas"] = schemas
            if self.default_dict:
                self.manager_input_params["default"] = self.default_dict
        elif self.default_string:
            self.manager_input_params["default"] = self.default_string

        if self.default_factory:
            self.manager_input_params["default_factory"] = self.default_factory

        self.manager_input_params["enable_inserts"] = self.insert
        self.manager_input_params["enable_deletes"] = self.delete
        if self.query_model:
            self.manager_input_params["query_model"] = self.query_model

        if self.query_limit < 0:
            msg = "Query limit must be non-negative"
            raise ValueError(msg)
        self.manager_input_params["query_limit"] = self.query_limit
        return self
