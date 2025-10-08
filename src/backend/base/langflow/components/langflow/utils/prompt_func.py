import re



def format_extraction_system_prompt(output_state, prompt_template):
    """Form the system prompt for structured output extraction. Use for extraction LLM when using AgentComponent."""
    verbose_str = output_state.schema.get("verbose_schema_str")

    # Replace {model_schema} with the actual verbose_str in system prompt (combined prompt if using pydantic)
    return prompt_template.replace("{langflow_model_schema}", verbose_str.strip())


def format_runtime_prompt(prompt, runtime):
    if not runtime.context:
        return prompt

    placeholders = re.findall(r"\{langflow_runtime_context\.([a-zA-Z_][a-zA-Z0-9_]*)\}", prompt)
    placeholders = list(set(placeholders))
    for placeholder in placeholders:
        if placeholder in runtime.context.__dict__:
            value = getattr(runtime.context, placeholder)
            # Convert to string, allowing for None, False, 0, empty strings
            prompt = prompt.replace(f"{{langflow_runtime_context.{placeholder}}}", str(value))
        else:
            msg = (
                f"Runtime context field '{placeholder}' not found in context. "
                f"Available fields: {list(runtime.context.__dict__.keys())}"
            )
            raise ValueError(msg)
    return prompt


def format_prompt(prompt_template, state, node_name) -> str:
    """Format the prompt by replacing placeholders with actual values from the state object.

    Use {{}} to escape literal braces in schema, and {} for variables.
    """
    # First, temporarily replace escaped braces {{}} with a placeholder
    temp_placeholder = "___ESCAPED_BRACE___"
    formatted_prompt = prompt_template.replace(
        "{{", f"{temp_placeholder}OPEN").replace("}}", f"{temp_placeholder}CLOSE"
        )

    # Extract variable placeholders (now won't match escaped braces)
    placeholders = re.findall(r"{([a-zA-Z_][a-zA-Z0-9_.]*?)}", formatted_prompt)
    placeholders = list(set(placeholders))  # Remove duplicates

    from pydantic import BaseModel
    for placeholder in placeholders:
        # Check if the placeholder exists as an attribute in the state object
        if "." not in placeholder:
            if not hasattr(state, placeholder):
                msg = (
                    f"{node_name} | Placeholder '{{{placeholder}}}' not found in the input state of node. "
                    f"Check your prompt template and ensure it matches the state attributes.\n"
                    f"Available attributes: [{', '.join(state.__dict__.keys())}]"
                )
                raise ValueError(msg)
            # Replace the placeholder with the actual value
            value = getattr(state, placeholder)
            formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", str(value))

        if "." in placeholder:
            var, field = placeholder.split(".")
            if not hasattr(state, var):
                msg = (
                    f"{node_name} | Variable '{var}' not found in the input state of node. "
                    f"Check your prompt template and ensure it matches the state attributes.\n"
                    f"Available attributes: [{', '.join(state.__dict__.keys())}]"
                )
                raise ValueError(msg)


            if isinstance(getattr(state, var), dict):
                if field not in getattr(state, var):
                    msg = (
                        f"{node_name} | Field '{field}' not found in the dictionary attribute '{var}'. "
                        f"Check your prompt template and ensure it matches the state attributes.\n"
                        f"Available fields in '{var}': [{', '.join(getattr(state, var).keys())}]"
                    )
                    raise ValueError(msg)
                # Replace the placeholder with the actual value
                value = getattr(state, var)[field]
                formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", str(value))

            if isinstance(getattr(state, var), BaseModel):
                if not hasattr(getattr(state, var), field):
                    msg = (
                        f"{node_name} | Field '{field}' not found in the model attribute '{var}'. "
                        f"Check your prompt template and ensure it matches the state attributes.\n"
                        f"Available fields in '{var}': [{', '.join(getattr(state, var).__dict__.keys())}]"
                    )
                    raise ValueError(msg)

                # Replace the placeholder with the actual value
                value = getattr(getattr(state, var), field)
                formatted_prompt = formatted_prompt.replace(f"{{{placeholder}}}", str(value))

    # Restore escaped braces
    return formatted_prompt.replace(f"{temp_placeholder}OPEN", "{").replace(f"{temp_placeholder}CLOSE", "}")


def format_all_prompts(prompt, mem_str, runtime, state, node_name):
    formatted_prompt = format_extraction_system_prompt(prompt)
    formatted_prompt = formatted_prompt.replace("{langflow_mem_data}", mem_str)
    formatted_prompt = format_runtime_prompt(formatted_prompt, runtime)
    return format_prompt(formatted_prompt, state, node_name)


def form_memory_str_for_prompt(memories_input) -> str:
    mem_ = set()
    # print("len of memories:", len(self.memories))
    for mem in memories_input:
        for m, m_format in mem.items():
            # print("Initial memory content:", m)
            # Extract variable placeholders (now won't match escaped braces)
            placeholders = re.findall(r"{([a-zA-Z_][a-zA-Z0-9_]*)}", m_format)

            m_val = getattr(m, "value", None)
            if not m_val:
                continue

            # Get content in mem_val
            m_val_content = m_val.get("content", None)
            # print(f"Raw memory content: {mem_val_content}")
            if isinstance(m_val_content, dict):
                for placeholder in placeholders:
                    if placeholder not in m_val_content:
                        print(f"Missing placeholder {{{placeholder}}} in memory. Using raw content.")  # noqa: T201
                        m_val_content = str(m_val_content)
                        # print(f"Formatted memory content as dict: {m_val_content}")
                        break
                if isinstance(m_val_content, dict):
                    # If still dict, format using all keys
                    m_val_content = m_format.format(**m_val_content)
            elif isinstance(m_val_content, str):
                # print(f"Formatted memory content as str: {m_val_content}")
                m_val_content = str(m_val_content)

            mem_.add(m_val_content.strip())
    return "\n".join(mem_)