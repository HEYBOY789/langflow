from langflow.base.agents.crewai.tasks import SequentialTask  # type: ignore  # noqa: N999, PGH003
from langflow.custom import Component
from langflow.io import BoolInput, DropdownInput, HandleInput, MultilineInput, Output


class SequentialTaskForLangGraphComponent(Component):
    display_name: str = "Sequential Task For LangGraph"
    description: str = ("This is used for LangGraph to interact with CrewAI's SequentialTask. "
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
            "• If using Pydantic, make sure you instruct the "
            "agent to return a json object that matches the output model schema.\n"
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
            display_name="Pydantic Output",
            input_types=["ModelClassWrapper"],
            info="Using this will output a pydantic class.",
            value=None,
            dynamic=True,
            show=True,
            required=True
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
        HandleInput(
            name="agent",
            display_name="Agent",
            input_types=["Agent"],
            info="CrewAI Agent that will perform the task",
            required=True,
        ),
        HandleInput(
            name="task",
            display_name="Task",
            input_types=["SequentialTask"],
            info="CrewAI Task that will perform the task",
        ),
        BoolInput(
            name="async_execution",
            display_name="Async Execution",
            value=True,
            advanced=True,
            info="Boolean flag indicating asynchronous task execution.",
        ),
    ]

    outputs = [
        Output(display_name="Task", name="task_output", method="build_task"),
    ]


    def _pre_run_setup(self):
        if self.output_type == "State Field":
            self.pydantic_model = None
        else:
            self.pydantic_model = self.pydantic_output.model_class


    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "output_type":
            if field_value == "State Field":
                build_config["pydantic_output"]["show"] = False
                build_config["pydantic_output"]["required"] = False
            else:
                build_config["pydantic_output"]["show"] = True
                build_config["pydantic_output"]["required"] = True
        return build_config

    def build_task(self) -> list[SequentialTask]:
        tasks: list[SequentialTask] = []
        task = SequentialTask(
            description=self.task_description,
            expected_output=self.expected_output,  # Now this is already a string
            tools=self.agent.tools,
            async_execution=False,
            agent=self.agent,
            output_pydantic=self.pydantic_model
        )
        tasks.append(task)
        self.status = task
        if self.task:
            if isinstance(self.task, list) and all(isinstance(task, SequentialTask) for task in self.task):
                tasks = self.task + tasks
            elif isinstance(self.task, SequentialTask):
                tasks = [self.task, *tasks]
        return tasks
