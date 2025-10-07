from langgraph.graph import StateGraph  # noqa: N999
from pydantic import BaseModel

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, Output


class CreateStateGraphComponent(Component):
    display_name = "State Graph For LangGraph"
    description = "Create State Graph for LangGraph"
    documentation: str = "https://langchain-ai.github.io/langgraph/concepts/low_level/#state"
    icon = "LangChain"
    name = "CreateStateGraph"

    inputs = [
        HandleInput(
            name="overall_state",
            display_name="Overall State Of Graph",
            info="Overall state of the graph. This is the public state shared across nodes.",
            input_types=["ModelClassWrapper"],
        ),
        HandleInput(
            name="input_state",
            display_name="Input State Of Graph",
            info="Input state for the graph. If not provided, overall_state will be used as the input state. "
            "Must match input state of first node",
            input_types=["ModelClassWrapper"],
        ),
        HandleInput(
            name="output_state",
            display_name="Output State Of Graph",
            info="Output state for the graph. If not provided, overall_state will be used as the output state. "
            "Must match input state of last node",
            input_types=["ModelClassWrapper"],
        ),
        HandleInput(
            name="context_schema",
            display_name="Context Schema",
            info="Connect structured data to generate a context schema for Graph's runtime configuration. "
            "If not provided, no runtime configuration will be applied.",
            input_types=["ModelClassWrapper"],
        ),
    ]

    outputs = [
        Output(display_name="Graph Entry", name="output", method="build_output"),
    ]

    def _get_model_class(self, state_wrapper) -> type[BaseModel]:
        """Extract model class from ModelClassWrapper."""
        if not state_wrapper:
            return None
        return state_wrapper.model_class

    def _validate_state_inputs(self) -> None:
        """Validate that we have the required state inputs."""
        if not self.overall_state and not (self.input_state and self.output_state):
            msg = "Either 'overall_state' or both 'input_state' and 'output_state' must be provided."
            raise ValueError(
                msg
            )

    def _setup_state_models(self) -> tuple[type[BaseModel], type[BaseModel], type[BaseModel]]:
        """Setup and return (overall_model, input_model, output_model).

        Returns:
            tuple: (overall_model, input_model, output_model)
        """
        if self.overall_state:
            return self._setup_with_overall_state()
        return self._setup_without_overall_state()

    def _setup_with_overall_state(self) -> tuple[type[BaseModel], type[BaseModel], type[BaseModel]]:
        """Setup state models when overall_state is provided."""
        overall_model = self._get_model_class(self.overall_state)

        # Determine input model
        if self.input_state:
            input_model = self._get_model_class(self.input_state)
            self.graph_input_state = self.input_state
        else:
            input_model = overall_model
            self.graph_input_state = self.overall_state

        # Determine output model
        if self.output_state:
            output_model = self._get_model_class(self.output_state)
            self.graph_output_state = self.output_state
        else:
            output_model = overall_model
            self.graph_output_state = self.overall_state

        return overall_model, input_model, output_model

    def _setup_without_overall_state(self) -> tuple[type[BaseModel], type[BaseModel], type[BaseModel]]:
        """Setup state models when overall_state is not provided."""
        input_model = self._get_model_class(self.input_state)
        output_model = self._get_model_class(self.output_state)

        # Create combined overall state from input and output
        class OverallState(input_model, output_model):
            """Dynamically created overall state combining input and output models."""

        overall_model = OverallState

        # Store in context
        self.graph_input_state = self.input_state
        self.graph_output_state = self.output_state

        return overall_model, input_model, output_model

    def _setup_context_schema(self) -> type[BaseModel]:
        """Setup context schema if provided."""
        self.graph_context_schema = self.context_schema
        if not self.context_schema:
            return None
        return self._get_model_class(self.context_schema)

    def _create_state_graph(
            self, overall_model: type[BaseModel],
            input_model: type[BaseModel],
            output_model: type[BaseModel],
            context_schema: type[BaseModel] | None = None) -> StateGraph:
        """Create and return the StateGraph instance."""
        return StateGraph(
            state_schema=overall_model,
            input_schema=input_model,
            output_schema=output_model,
            context_schema=context_schema
        )

    def _store_context(self, builder: StateGraph) -> None:
        """Store the graph builder in context for nodes to access."""
        self.graph_builder = builder

    def build_output(self) -> "CreateStateGraphComponent":
        """Build and configure the state graph.

        Returns:
            CreateStateGraphComponent: Self for chaining
        """
        # Validate inputs
        self._validate_state_inputs()

        # Setup state models
        overall_model, input_model, output_model = self._setup_state_models()

        # Setup context schema if provided
        context_schema = self._setup_context_schema()

        # Create state graph
        builder = self._create_state_graph(overall_model, input_model, output_model, context_schema)

        # Store in context
        self._store_context(builder)

        # Log successful creation
        self._log_graph_creation(overall_model, input_model, output_model)

        return self

    def _log_graph_creation(
            self, overall_model: type[BaseModel],
            input_model: type[BaseModel],
            output_model: type[BaseModel]) -> None:
        """Log information about the created graph."""
        print("✓ StateGraph created successfully:")  # noqa: T201
        print(f"  - Overall State: {overall_model.__name__}")  # noqa: T201
        print(f"  - Input State: {input_model.__name__}")  # noqa: T201
        print(f"  - Output State: {output_model.__name__}")  # noqa: T201

    def get_graph_info(self) -> dict:
        """Get information about the configured graph.

        Returns:
            dict: Graph configuration information
        """
        return {
            "has_overall_state": bool(self.overall_state),
            "has_input_state": bool(self.input_state),
            "has_output_state": bool(self.output_state),
            "overall_state_name": getattr(self.overall_state, "schema", {}).get("class_name")
            if self.overall_state else None,
            "input_state_name": getattr(self.input_state, "schema", {}).get("class_name")
            if self.input_state else None,
            "output_state_name": getattr(self.output_state, "schema", {}).get("class_name")
            if self.output_state else None,
        }
