from collections.abc import Callable

from loguru import logger

from langflow.custom.custom_component.component import Component
from langflow.custom.utils import get_function
from langflow.io import CodeInput, Output
from langflow.schema.dotdict import dotdict


class CreateRetryPolicyFunctionForLangGraph(Component):
    display_name = "Retry Policy Function For LangGraph"
    description = "Define and create a retry policy function for LangGraph nodes."
    documentation = "https://langchain-ai.github.io/langgraph/concepts/low_level/#retries"
    icon = "LangChain"
    name = "CreateRetryPolicyFunction"

    inputs = [
        CodeInput(
            name="function_code",
            display_name="Function Code",
            info="The code for the retry policy function. Must return a RetryPolicy object.",
            value="""from langgraph.types import RetryPolicy

def get_retry_policy():
    \"\"\"Create a retry policy for database operations\"\"\"
    return RetryPolicy(max_attempts=5)""",
        ),
    ]

    outputs = [
        Output(
            name="retry_policy_output",
            display_name="Retry Policy",
            method="get_retry_policy_output",
        ),
    ]

    def get_function_callable(self) -> Callable:
        function_code = self.function_code
        self.status = function_code
        return get_function(function_code)

    def execute_function(self) -> list[dotdict | str] | dotdict | str:
        function_code = self.function_code

        if not function_code:
            return "No function code provided."

        try:
            func = get_function(function_code)
            return func()
        except (ValueError, TypeError, ImportError) as e:  # More specific exceptions
            logger.opt(exception=True).debug("Error executing function")
            return f"Error executing function: {e}"

    def get_retry_policy_output(self) -> "CreateRetryPolicyFunctionForLangGraph":
        results = self.execute_function()
        results = results if isinstance(results, list) else [results]
        return results[0]
