from langflow.custom.custom_component.component import Component  # noqa: N999
from langflow.io import CodeInput, Output


class CreateReducerFunctionForLangGraph(Component):
    display_name = "Reducer Function For LangGraph"
    description = "Define and create a reducer function for LangGraph."
    documentation= "https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers"
    icon = "LangChain"
    name = "CreateReducerFunction"

    inputs = [
        CodeInput(
            name="function_code",
            display_name="Function Code",
            info="The code for the function.",
            value="def node_1(a, b):\n"
                    '    """Same as a + b."""\n'
                    "    return a + b"
        ),
    ]

    outputs = [
        Output(
            name="function_output",
            display_name="Custom Reducer Functions",
            method="get_function_callable",
        ),
    ]

    def get_function_callable(self) -> "CreateReducerFunctionForLangGraph":
        function_code = self.function_code

        # Create a local namespace to execute the code
        local_namespace = {}

        try:
            # Execute the function code in the local namespace
            exec(function_code, globals(), local_namespace)  # noqa: S102

            # Extract function name from the code
            func_name = function_code.split("def ")[1].split("(")[0].strip()

            # Get the actual function object
            function_obj = local_namespace[func_name]
        except (SyntaxError, NameError, KeyError, IndexError) as e:
            msg = f"Error executing function code: {e}"
            raise ValueError(msg) from e
        else:
            return {func_name: function_obj}
