import ast  # noqa: N999
import contextlib
import importlib
from collections.abc import Callable

from lfx.custom.custom_component.component import Component
from lfx.io import CodeInput, Output


class PythonFunctionComponentForLangGraph(Component):
    display_name = "Python Function For LangGraph"
    description = "Define a Python function from code for a node in langgraph."
    icon = "Python"
    name = "PythonFunction"
    legacy = True

    inputs = [
        CodeInput(
            name="function_code",
            display_name="Function Code",
            info="The code for the function.",
        ),
    ]

    outputs = [
        Output(
            name="function_output",
            display_name="Function Callable",
            method="get_function_callable",
        ),
    ]

    def get_function_callable(self) -> Callable:
        function_code = self.function_code
        self.status = function_code
        return self.get_function(function_code)

    def extract_function_name(self, code):
        module = ast.parse(code)
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        msg = "No function definition found in the code string"
        raise ValueError(msg)

    def get_function(self, code):
        """Get the function."""
        function_name = self.extract_function_name(code)

        return self.create_function(code, function_name)

    def create_function(self, code, function_name):
        if not hasattr(ast, "TypeIgnore"):

            class TypeIgnore(ast.AST):
                _fields = ()

            ast.TypeIgnore = TypeIgnore

        module = ast.parse(code)
        exec_globals = globals().copy()

        for node in module.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    try:
                        if isinstance(node, ast.ImportFrom):
                            module_name = node.module
                            exec_globals[alias.asname or alias.name] = getattr(
                                importlib.import_module(module_name), alias.name
                            )
                        else:
                            module_name = alias.name
                            exec_globals[alias.asname or alias.name] = importlib.import_module(module_name)
                    except ModuleNotFoundError as e:
                        msg = f"Module {alias.name} not found. Please install it and try again."
                        raise ModuleNotFoundError(msg) from e

        function_code = next(
            node for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
            node.name == function_name
        )
        function_code.parent = None
        code_obj = compile(ast.Module(body=[function_code], type_ignores=[]), "<string>", "exec")
        exec_locals = dict(locals())
        with contextlib.suppress(Exception):
            exec(code_obj, exec_globals, exec_locals)  # noqa: S102
        exec_globals[function_name] = exec_locals[function_name]

        # Return a function that imports necessary modules and calls the target function
        def wrapped_function(*args, **kwargs):
            for module_name, module in exec_globals.items():
                if isinstance(module, type(importlib)):
                    globals()[module_name] = module

            return exec_globals[function_name](*args, **kwargs)

        return wrapped_function
