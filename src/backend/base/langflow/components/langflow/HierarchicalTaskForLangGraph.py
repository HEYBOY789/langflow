from langflow.base.agents.crewai.tasks import HierarchicalTask  # type: ignore  # noqa: N999, PGH003
from langflow.custom import Component
from langflow.io import DropdownInput, HandleInput, MultilineInput, Output


class HierarchicalTaskForLangGraphComponent(Component):
    display_name: str = "Hierarchical Task For LangGraph"
    description: str = ("This is used for LangGraph to interact with CrewAI's HierarchicalTask. "
    "Each task must have a description, an expected output and an agent responsible for execution. "
    "For the prompt, use {variable} to refer to the state's fields. "
    "The 'variable' should match the fields in the Input State Of The Node. ")
    icon = "CrewAI"
    inputs = [
        MultilineInput(
            name="task_description",
            display_name="Description",
            info="Descriptive text detailing task's purpose and execution.",
        ),
        MultilineInput(
            name="expected_output",
            display_name="Expected Output",
            info="Clear definition of expected task outcome.\n"
            "• If using Pydantic, make sure you instruct the agent to"
            " return a json object that matches the output model schema.\n"
            'Example: Only return a json object with the following structure: {"field_name": "value"}. Nothing more.\n',
        ),
        DropdownInput(
            name="output_type",
            display_name="Output Pydantic or a State Field",
            info="Select the output type. Must match the output type of it's graph node.",
            options=["Pydantic", "State Field"],
            value="Pydantic",
            real_time_refresh=True,
        ),
        HandleInput(
            name="pydantic_output",
            display_name="Pydantic Model",
            input_types=["ModelClassWrapper"],
            info="Using this will output a pydantic class.",
            value=None,
            show=True,
            required=True,
            dynamic=True
        ),
        HandleInput(
            name="tools",
            display_name="Tools",
            input_types=["Tool"],
            is_list=True,
            info="List of tools/resources limited for task execution. Uses the Agent tools by default.",
            required=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Task", name="task_output", method="build_task"),
    ]


    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "output_type":
            if field_value == "State Field":
                build_config["pydantic_output"]["show"] = False
                build_config["pydantic_output"]["required"] = False
            else:
                build_config["pydantic_output"]["show"] = True
                build_config["pydantic_output"]["required"] = True
        return build_config


    def _pre_run_setup(self):
        if self.output_type == "State Field":
            self.pydantic_model = None
        else:
            self.pydantic_model = self.pydantic_output.model_class


    def build_task(self) -> HierarchicalTask:
        task = HierarchicalTask(
            description=self.task_description,
            expected_output=self.expected_output,
            tools=self.tools or [],
            output_pydantic=self.pydantic_model
        )
        self.status = task
        return task
