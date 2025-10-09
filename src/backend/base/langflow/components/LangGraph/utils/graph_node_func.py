from langgraph.graph import (  # noqa: INP001
    END,
    START,
)


def build_params_for_add_node(node_caching_input, retry_policy_input, defer_node_input):

    # Add this node to the builder FIRST
    if node_caching_input > 0:
        from langgraph.types import CachePolicy
        caching_ = CachePolicy(ttl=node_caching_input)
    else:
        caching_ = None

    # Add retry policy if provided
    policy_ = retry_policy_input or None

    return {
        "cache_policy": caching_,
        "retry_policy": policy_,
        "defer": defer_node_input
    }

def detect_and_register_edges(builder, node_name, previous_nodes):
    """Detect edges based on previous_nodes connections and add them directly to builder."""
    if not builder:
        msg = f"{node_name} | No StateGraph builder found in context."
        raise ValueError(msg)

    # Create edges from previous nodes to this node (if any)
    if previous_nodes:
        # If previous_node is conditonal edge,
        # just pass because all the logic is handled in ConditionalEdgeForLangGraph
        for prev_node_component in previous_nodes:
            # Create a start node if previous node is from CreateStateGraphComponent
            if prev_node_component.__class__.__name__ == "CreateStateGraphComponent":
                builder.add_edge(START, node_name)
                print(f"Added START edge: START -> {node_name}")  # noqa: T201
                continue

            # If previous_node is conditonal edge, Send API or Node that return Command,
            # just pass because all the logic is handled in ConditionalEdgeForLangGraph
            if prev_node_component.__class__.__name__ in [
                "ConditionalEdgeForLangGraph",
                "SendMapReduceForLangGraph",
                "GraphNodeForAgentWithCommand",
                "GraphNodeForCrewAIAgentWithCommand",
                "GraphNodeForCrewAICrewWithCommand",
                "GraphNodeForFunctionWithCommand",
                "GraphNodeAsSubGraphWithCommand"
            ]:
                print(f"Skipping {prev_node_component.__class__.__name__}")  # noqa: T201
                continue

            # Get the node name from the previous GraphNode component
            prev_node_name = prev_node_component.node_name
            builder.add_edge(prev_node_name, node_name)
            if node_name == END:
                print(f"Added END edge: {prev_node_name} -> END")  # noqa: T201
            else:
                print(f"Added edge: {prev_node_name} -> {node_name}")  # noqa: T201
    return builder


def check_if_field_is_list(state, field_name: str) -> bool:
    """Check if a field is a list type using the schema."""
    schema_fields = state.schema.get("fields", [])

    for field in schema_fields:
        if field["name"] == field_name:
            field_type_str = field["type"]
            # Check if the type string contains "List"
            return "List[" in field_type_str or field_type_str == "list"
    return False
