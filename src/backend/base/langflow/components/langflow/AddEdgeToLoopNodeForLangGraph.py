import logging  # noqa: N999

from langflow.custom.custom_component.component import Component
from langflow.io import HandleInput, MessageTextInput, Output

logger = logging.getLogger(__name__)

class AddEdgeToLoopNodeForLangGraph(Component):
    display_name = "Add Edge To Loop Node"
    description = (
        "Use this component to create an edge from a node that's already connected "
        "(directly or indirectly) to the node linked to this component. "
        "This is especially helpful for forming loops in the graph.\n"
        "Example:\n"
        "Node A → Conditional Edge → Node B → Node A\n"
        "Node B → End\n"
        "In this example, we want to connect Node B back to Node A to form a loop. "
        "Since Langflow doesn't natively support this, you can use this component to achieve it."
    )
    documentation: str = "https://langchain-ai.github.io/langgraph/how-tos/graph-api/#create-and-control-loops"
    icon = "LangChain"
    name = "AddEdgeToLoopNodeForLangGraph"

    inputs = [
        HandleInput(
            name="start_nodes",
            display_name="Start Nodes Of Loop Edge",
            info="Connect GraphNodes to create loop edges for those nodes.",
            input_types=[
                "GraphNodeForCrewAIAgent",
                "GraphNodeForAgent",
                "GraphNodeForCrewAICrew",
                "GraphNodeForFunction",
            ],
            required=True,
            is_list=True,
            # Ve sau co the xem xet them vao cac lua chon input.
            # Nhung ma khi nao gap toi truong hop do luc lam thuc te thi se them
        ),
        MessageTextInput(
            name="end_node",
            display_name="End Node Of Loop Edge",
            info="Specify the name of the end node.",
            required=True,
            placeholder="Example: Node A",
        ),
    ]

    outputs = [
        Output(display_name="Loop", name="output", method="add_loop_edge"),
    ]

    def _pre_run_setup(self):
        self.graph_input_state = self.start_nodes[0].graph_input_state
        self.graph_builder = self.start_nodes[0].graph_builder
        self.graph_output_state = self.start_nodes[0].graph_output_state

    def add_loop_edge(self) -> "AddEdgeToLoopNodeForLangGraph":
        # Get builder from shared data
        builder = self.graph_builder
        start_node_names = [node.node_name for node in self.start_nodes]
        builder.add_edge(start_node_names, self.end_node)
        logger.debug("Added edge: %s -> %s", start_node_names, self.end_node)
        return self
