from crewai import Crew, Process  # noqa: I001, N999

from langflow.base.agents.crewai.crew import BaseCrewComponent, Output, convert_llm # type: ignore  # noqa: PGH003
from langflow.io import DropdownInput, HandleInput


class HierarchicalCrewForLangGraphComponent(BaseCrewComponent):
    display_name: str = "Hierarchical Crew For LangGraph"
    description: str = (
        "Defines a group of agents, including their collaboration strategy and assigned tasks.\n"
        "NOTE: Make sure to enable verbose mode to track which task was executed last based on the agent's response. "
        "This is important for linking a Pydantic model in the final task so it can produce a structured output.\n"
        "Due to inconsistencies in how some models perform tasks, "
        "extensive testing may be required to achieve reliable results."
    )
    documentation: str = "https://docs.crewai.com/how-to/Hierarchical/"
    icon = "CrewAI"

    inputs = [
        *BaseCrewComponent._base_inputs,
        HandleInput(name="agents", display_name="Agents", input_types=["Agent"], is_list=True),
        HandleInput(name="tasks", display_name="Tasks", input_types=["HierarchicalTask"], is_list=True),
        DropdownInput(
            name="manager_type",
            display_name="Manager Type",
            options=["Manager LLM", "Manager Agent"],
            value="Manager Agent",
            info="Select the type of manager for the crew",
            required=True,
            real_time_refresh=True,
        ),
        HandleInput(
            name="llm",
            display_name="Manager Language Model",
            info="Manager LLM for the crew.",
            input_types=["LanguageModel"],
            required=False,
            dynamic=True,
            show=False
        ),
        HandleInput(
            name="manager_agent",
            display_name="Manager Agent",
            input_types=["Agent"],
            required=True,
            dynamic=True,
            show=True
        ),
    ]

    outputs = [
        Output(display_name="Crew", name="crew", method="build_crew"),
    ]

    def _pre_run_setup(self):
        # Ensure verbose is a boolean
        if isinstance(self.verbose, int):
            self.verbose = bool(self.verbose)

        if self.manager_agent:
            self.manager = {"manager_agent": self.manager_agent}
        else:
            manager_llm = convert_llm(self.llm)
            self.manager = {"manager_llm": manager_llm}


    def update_build_config(self, build_config, field_value, field_name = None):
        if field_name == "manager_type":
            if field_value == "Manager LLM":
                build_config["llm"]["required"] = True
                build_config["llm"]["show"] = True
                build_config["manager_agent"]["required"] = False
                build_config["manager_agent"]["show"] = False
            elif field_value == "Manager Agent":
                build_config["llm"]["required"] = False
                build_config["llm"]["show"] = False
                build_config["manager_agent"]["required"] = True
                build_config["manager_agent"]["show"] = True
        return build_config

    def build_crew(self) -> Crew:
        tasks, agents = self.get_tasks_and_agents()
        return Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical,
            verbose=self.verbose,
            memory=self.memory,
            cache=self.use_cache,
            max_rpm=self.max_rpm,
            share_crew=self.share_crew,
            function_calling_llm=self.function_calling_llm,
            **self.manager,
            step_callback=self.get_step_callback(),
            task_callback=self.get_task_callback(),
        )
