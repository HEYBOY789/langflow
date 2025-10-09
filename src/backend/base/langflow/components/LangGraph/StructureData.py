import sys  # noqa: N999
import types
from operator import add
from typing import Annotated, Optional, Union, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field

from langflow.custom import Component
from langflow.io import HandleInput, MessageTextInput, Output, TableInput
from langflow.schema.table import EditMode, FormatterType


class ModelClassWrapper:
        def __init__(self, model_class, schema):
            self.model_class = model_class
            self.schema = schema
            self._field_types = None  # Cache for field types

        def __str__(self):
            return f"ModelClass({self.schema['class_name']})"

        def create_instance(self, **kwargs):
            """Create an instance of the model class with the provided data."""
            return self.model_class(**kwargs)

        def to_dict(self):
            """Return a serializable representation."""
            return self.schema

        def get_verbose_schema(self):
            """Get the verbose schema string for AI prompts."""
            return self.schema.get("verbose_schema_str", "")

        def get_field_types(self):
            """Extract field types from model_class and return as dict mapping field names to types."""
            if self._field_types is not None:
                return self._field_types

            def extract_innermost_type(field_type):
                """Recursively extract the innermost actual type from nested annotations."""
                # Handle Annotated types first (strip annotations like operator.add)
                if get_origin(field_type) is Annotated:
                    # Get the base type (first argument of Annotated) and recurse
                    base_type = get_args(field_type)[0]
                    return extract_innermost_type(base_type)

                # Handle Optional/Union types
                if get_origin(field_type) is Union:
                    args = get_args(field_type)
                    # If it's Optional[Type] (Union[Type, None])
                    optional_args_count = 2
                    if len(args) == optional_args_count and type(None) in args:
                        # Get the non-None type and recurse
                        actual_type = args[0] if args[1] is type(None) else args[1]
                        return extract_innermost_type(actual_type)
                    # For other Union types, return as is (could be improved if needed)
                    return field_type

                # Handle List types
                if get_origin(field_type) is list:
                    # Get the inner type and recurse
                    inner_type = get_args(field_type)[0]
                    return extract_innermost_type(inner_type)

                # Base case: return the actual type
                return field_type

            type_hints = get_type_hints(self.model_class)
            field_types = {}

            for field_name, field_type in type_hints.items():
                # Extract the innermost actual type
                innermost_type = extract_innermost_type(field_type)
                field_types[field_name] = innermost_type

            # Cache the result
            self._field_types = field_types
            return field_types

class StructureData(Component):
    display_name = "Structure Data"
    description = "Use this to create structure data with pydantic"
    icon = "LangChain"
    name = "StructureData"

    inputs = [
        MessageTextInput(
            name="class_name",
            display_name="Class Name",
            info="Name of the class to be created.",
            required=True,
            value="MyClass",
        ),
        MessageTextInput(
            name="doc_string",
            info="Docstring of the class to be created. Must have when extracting Pydantic Output for better output",
            display_name="Docstring",
            advanced=True
        ),
        HandleInput(
            name="nested_models",
            display_name="Structure Data Classes",
            info="Models to be used as types in this model",
            input_types=["ModelClassWrapper"],
            is_list=True,
        ),
        HandleInput(
            name="custom_reducer_functions",
            display_name="Custom Reducer Functions",
            input_types=["CreateReducerFunctionForLangGraph"],
            info="A custom function to use with langraph's reducer feature (https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)",
            is_list=True,
            advanced=True
        ),
        TableInput(
            name="output_schema",
            display_name="Output Schema",
            info="Define the structure and data types for the model's output.",
            required=True,
            table_schema=[
                {
                    "name": "name",
                    "display_name": "Name",
                    "type": "str",
                    "description": "Specify the name of the output field.",
                    "default": "field",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "description",
                    "display_name": "Description",
                    "type": "str",
                    "description": "Describe the purpose of the output field.",
                    "default": "description of field",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "type",
                    "display_name": "Type",
                    "type": "str",
                    "edit_mode": EditMode.INLINE,
                    "description": "Indicate the data type of the output field "
                                  "(e.g., str, int, float, bool, dict, or custom model name).",
                    "options": ["str", "int", "float", "bool", "dict", "custom"],
                    "default": "str",
                },
                {
                    "name": "custom_type",
                    "display_name": "Custom Type Name",
                    "type": "str",
                    "description": "If type is 'custom', specify the name of the custom class",
                    "default": "",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "using_reducer",
                    "display_name": "Using Reducer",
                    "type": "str",
                    "description": "Indicate if the field is using a reducer. This should only be used for LangGraph",
                    "default": "None",
                    "edit_mode": EditMode.INLINE,
                    "options": ["None", "Default Operator Add", "Custom Reducer Function"]
                },
                {
                    "name": "custom_reducer_function_name",
                    "display_name": "Custom Reducer Function Name",
                    "type": "str",
                    "description": "If using a custom reducer function, specify the name of the function.",
                    "default": "",
                    "edit_mode": EditMode.INLINE,
                },
                # {
                #     "name": "multiple",
                #     "display_name": "Multiple",
                #     "type": FormatterType.boolean,
                #     "description": "Indicate if the field can have multiple values (e.g., a list of strings).",
                #     "default": False,
                #     "edit_mode": EditMode.INLINE,
                # },
                # {
                #     "name": "required",
                #     "display_name": "Required",
                #     "type": FormatterType.boolean,
                #     "description": "Indicate if the field is required in the model.",
                #     "default": True,
                #     "edit_mode": EditMode.INLINE,
                # }
                {
                    "name": "multiple",
                    "display_name": "Multiple",
                    "type": FormatterType.boolean,
                    "description": "Indicate if the field can have multiple values (e.g., a list of strings).",
                    "options": ["True", "False"],
                    "default": "False",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "required",
                    "display_name": "Required",
                    "type": FormatterType.boolean,
                    "description": "Indicate if the field is required in the model.",
                    "options": ["True", "False"],
                    "default": "True",
                    "edit_mode": EditMode.INLINE,
                }
            ],
            value=[
                # {
                #     "name": "field",
                #     "description": "description of field",
                #     "type": "str",
                #     "using_reducer": "None",
                #     "multiple": False,
                #     "required": True,
                # },
                {
                    "name": "field",
                    "description": "description of field",
                    "type": "str",
                    "using_reducer": "None",
                    "multiple": "False",
                    "required": "True",
                },
            ],
        )
    ]

    outputs = [
        Output(display_name="Pydantic Model", name="output", method="create_model_class"),
    ]

    def verbose_print(self, namespace: dict):
        verbose_str = ""
        # If nested models are provided, include them in the verbose output:
        if self.nested_models:
            for model_wrapper in self.nested_models:
                nested_verbose_str = model_wrapper.schema.get("verbose_str")
                if nested_verbose_str not in verbose_str:
                    verbose_str += nested_verbose_str + "\n"

        verbose_str += f"class {self.class_name}(BaseModel):\n"
        for field_name, (field_type, field_obj) in namespace.items():
            type_name = self._get_clean_type_name(field_type)
            verbose_str += f'    {field_name}: {type_name} = Field(description="{field_obj.description}")\n'
        verbose_str += ""
        print(verbose_str)  # noqa: T201
        return verbose_str


    def _get_clean_type_name(self, field_type) -> str:
        """Get a clean, readable type name without operator.add annotations."""
        # Handle Optional (Union[type, None]) FIRST - before checking for Annotated
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            # print("Debug: Found Union type")
            # Check if it's Optional (Union with None)
            args = field_type.__args__
            optional_args_count = 2
            if len(args) == optional_args_count and type(None) in args:
                # print("Debug: This is Optional")
                # Find the non-None type
                inner_type = args[0] if args[1] is type(None) else args[1]
                inner_type_name = self._get_clean_type_name(inner_type)
                return f"Optional[{inner_type_name}]"
            #print("Debug: This is regular Union")
            # Regular Union
            type_names = [self._get_clean_type_name(arg) for arg in args if arg is not type(None)]
            return f"Union[{', '.join(type_names)}]"

        # Handle Annotated types (strip operator.add) - check by __origin__ first
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Annotated:
            # print("Debug: Found Annotated type (by origin check), extracting base type")
            # Get the base type (first argument of Annotated)
            base_type = field_type.__args__[0]
            # print(f"Debug: Base type is: {base_type}")
            return self._get_clean_type_name(base_type)

        # Fallback: check if this is an Annotated type by checking the string representation
        # This is for cases where __origin__ check doesn't work
        type_str = str(field_type)
        if (("typing.Annotated[" in type_str or "Annotated[" in type_str)
                and hasattr(field_type, "__args__")
                and field_type.__args__):
            # print("Debug: Found Annotated type (by string check), extracting base type")
            # Get the base type (first argument of Annotated)
            base_type = field_type.__args__[0]
            # print(f"Debug: Base type is: {base_type}")
            return self._get_clean_type_name(base_type)

        # Handle List
        if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
            # print("Debug: Found List type")
            inner_type = field_type.__args__[0]
            inner_type_name = self._get_clean_type_name(inner_type)
            return f"List[{inner_type_name}]"

        # Handle basic types
        if hasattr(field_type, "__name__"):
            # print(f"Debug: Found basic type with __name__: {field_type.__name__}")
            return field_type.__name__

        # Handle custom model classes - check if it's a Pydantic model
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # print(f"Debug: Found Pydantic model: {field_type.__name__}")
            return field_type.__name__

        # Handle custom model classes (alternative check)
        if hasattr(field_type, "__class__") and hasattr(field_type.__class__, "__name__"):
            # print(f"Debug: Found class with __class__.__name__: {field_type.__class__.__name__}")
            return field_type.__class__.__name__

        # Fallback - this should not happen with properly constructed types
        print(f"Debug: Fallback to str(): {field_type!s}")  # noqa: T201
        return str(field_type)

    def build_verbose_schema(self, namespace: dict, custom_models: dict) -> str:
        """Build a verbose schema format for AI prompts that shows field structure with descriptions."""

        def build_model_schema(model_class, custom_models):
            """Recursively build schema for a Pydantic model."""
            if not hasattr(model_class, "model_fields"):
                return None

            schema = {}
            for field_name, field_info in model_class.model_fields.items():
                field_type = field_info.annotation
                description = (field_info.description
                               if hasattr(field_info, "description")
                               else f"description of {field_name}")

                # Handle the field recursively
                field_schema = get_field_schema_recursive(field_type, description, custom_models)
                schema[field_name] = field_schema

            return schema

        def get_field_schema_recursive(field_type, description, custom_models):
            """Recursively get schema for a field, handling any level of nesting."""
            # Handle Annotated types first - extract base type
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Annotated:
                base_type = field_type.__args__[0]
                return get_field_schema_recursive(base_type, description, custom_models)
            # Handle Optional types - extract inner type
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
                args = field_type.__args__
                optional_args_count = 2
                if len(args) == optional_args_count and type(None) in args:
                    # This is Optional
                    inner_type = args[0] if args[1] is type(None) else args[1]
                    return get_field_schema_recursive(inner_type, description, custom_models)
                # Regular Union - just return description with type
                type_name = self._get_clean_type_name(field_type)
                return f"{description} (type: {type_name})"

            # Handle List types
            if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                inner_type = field_type.__args__[0]

                # Check if it's a list of custom models
                if inner_type in custom_models.values() or (hasattr(inner_type, "model_fields")):
                    # Build nested schema for custom model and wrap in list
                    nested_schema = build_model_schema(inner_type, custom_models)
                    if nested_schema:
                        return [nested_schema]  # Return as a list to show it's a list of objects
                    type_name = self._get_clean_type_name(field_type)
                    return f"{description} (type: {type_name})"
                # List of basic types
                type_name = self._get_clean_type_name(field_type)
                return f"{description} (type: {type_name})"

            # Handle direct custom model types
            if field_type in custom_models.values() or (hasattr(field_type, "model_fields")):
                # Build nested schema for custom model
                nested_schema = build_model_schema(field_type, custom_models)
                if nested_schema:
                    return nested_schema
                type_name = self._get_clean_type_name(field_type)
                return f"{description} (type: {type_name})"

            # Handle basic types
            type_name = self._get_clean_type_name(field_type)
            return f"{description} (type: {type_name})"

        def get_field_schema(field_type, field_obj):
            """Get schema for a single field - wrapper for backward compatibility."""
            description = field_obj.description
            return get_field_schema_recursive(field_type, description, custom_models)

        # Build the main schema
        schema_dict = {}
        for field_name, (field_type, field_obj) in namespace.items():
            schema_dict[field_name] = get_field_schema(field_type, field_obj)

        print(str(schema_dict))  # noqa: T201

        return str(schema_dict)

    def create_model_class(self) -> ModelClassWrapper:
        type_map = {
            "str": str,
            "float": float,
            "int": int,
            "bool": bool,
            "dict": dict,
        }

        # Get available custom models from nested_models
        custom_models = {}
        custom_model_names = []
        if self.nested_models:
            for model_wrapper in self.nested_models:
                if hasattr(model_wrapper, "model_class") and hasattr(model_wrapper, "schema"):
                    class_name = model_wrapper.schema.get("class_name", "")
                    if class_name:
                        custom_models[class_name] = model_wrapper.model_class
                        custom_model_names.append(class_name)

        # Print available custom types to help the user
        if custom_model_names:
            print(f"Available custom types: {', '.join(custom_model_names)}")  # noqa: T201

        # Get available custom reducer functions from custom_reducer_functions
        custom_reducer_functions_ = {}
        custom_reducer_function_names = []
        if self.custom_reducer_functions:
            for item in self.custom_reducer_functions:
                for name, func in item.items():
                    custom_reducer_functions_[name] = func
                    custom_reducer_function_names.append(name)

        # Print available custom reducer functions to help the user
        if custom_reducer_function_names:
            print(f"Available custom reducer functions: {', '.join(custom_reducer_function_names)}")  # noqa: T201

        namespace = {}

        for f in self.output_schema:
            field_name = f["name"]
            description = f["description"]
            type_str = f["type"]
            custom_type = f.get("custom_type", "")
            using_reducer = f.get("using_reducer", "None")
            custom_reducer = f.get("custom_reducer_function_name", "")
            # multiple = f.get("multiple", False)
            # required = f.get("required", True)
            multiple = f.get("multiple", "False")
            required = f.get("required", "True")

            # Enhanced error handling for custom types
            if type_str == "custom":
                if not custom_type:
                    msg = (f"Field '{field_name}' has type 'custom' but no custom_type specified. "
                           f"Available types: {custom_model_names}")
                    raise ValueError(msg)
                if custom_type not in custom_models:
                    msg = (f"Custom type '{custom_type}' for field '{field_name}' not found. "
                           f"Available types: {custom_model_names}")
                    raise ValueError(msg)

                base_type = custom_models[custom_type]
            else:
                base_type = type_map.get(type_str, str)

            # Add List
            # annotated_type = list[base_type] if multiple else base_type
            annotated_type = list[base_type] if multiple == "True" else base_type

            # Add Optional
            # if required:
            #     # Keep annotated_type as is
            #     field_kwargs = {"description": description}
            # else:
            #     annotated_type = Optional[annotated_type]
            #     field_kwargs = {"description": description, "default": None}
            if required == "True":
                # Keep annotated_type as is
                field_kwargs = {"description": description}
            else:
                annotated_type = Optional[annotated_type]  # noqa: UP045
                field_kwargs = {"description": description, "default": None}

            # Add Reducer
            if using_reducer == "Default Operator Add":
                final_type = Annotated[annotated_type, add]
            elif using_reducer == "Custom Reducer Function":
                # Find function with name
                if not custom_reducer:
                    msg = (f"Field '{field_name}' using 'Custom Reducer Function' "
                           f"but no custom_reducer_function_name specified.")
                    raise ValueError(msg)
                if custom_reducer not in custom_reducer_functions_:
                    msg = (f"Custom reducer function '{custom_reducer}' for field '{field_name}' "
                           f"not found. Available functions: {custom_reducer_function_names}")
                    raise ValueError(msg)
                custom_reducer = custom_reducer_functions_[custom_reducer]
                # Set reducer
                final_type = Annotated[annotated_type, custom_reducer]
            elif using_reducer == "None":
                final_type = annotated_type

            print("Debug: ", final_type)  # noqa: T201
            namespace[field_name] = (final_type, Field(**field_kwargs))

        # Create a safe class name for pickling (remove spaces, special chars)
        safe_class_name = self.class_name.replace(" ", "").replace("-", "_").replace(".", "_")

        # Use the actual module where this component exists
        current_module = self.__class__.__module__  # Gets 'langflow.components.langgraph.StructureData'
        # Create the class with proper module attribution for pickling
        default_docstring = f"A Pydantic model representing {self.class_name} data structure."
        class_docstring = default_docstring if not self.doc_string else self.doc_string

        class_dict = {
            "__annotations__": {k: v[0] for k, v in namespace.items()},
            "__doc__": class_docstring,
            "__module__": current_module,  # Proper module for pickling
            "__qualname__": safe_class_name,
            **{k: v[1] for k, v in namespace.items()}
        }

        # Create the model class
        model_class = type(safe_class_name, (BaseModel,), class_dict)
        # Register the class in sys.modules for pickling support
        module_name = "langflow.components.langgraph.dynamic_models"
        if module_name not in sys.modules:
            sys.modules[module_name] = types.ModuleType(module_name)

        # Set the class in the module so it can be pickled
        setattr(sys.modules[module_name], safe_class_name, model_class)

        # Ensure the model's module reference is correct
        model_class.__module__ = module_name

        schema = {
            "class_name": self.class_name,
            "fields": [{
                "name": field_name,
                "type": str(field_type[0]),
                "description": field_type[1].description,
                "is_custom": isinstance(field_type[0], type) and issubclass(field_type[0], BaseModel)
            } for field_name, field_type in namespace.items()],
            "verbose_str": self.verbose_print(namespace),
            "verbose_schema_str": self.build_verbose_schema(namespace, custom_models)
        }
        return ModelClassWrapper(model_class, schema)
