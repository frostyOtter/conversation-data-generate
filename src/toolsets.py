import yaml
import random
from typing import Optional, Tuple, Dict
from src.conversation_models import ToolCallIO


def read_tools_registry_from_yaml_file(file_path: str) -> Tuple[Dict, Dict]:
    with open(file_path, "r") as file:
        file_content = yaml.safe_load(file)

    return (
        file_content.get("TOOLS_REGISTRY"),
        file_content.get("TOPIC_KEYWORD_TO_TOOLS"),
    )


def generate_mock_tool_call(tool_name: str, registry: dict) -> Optional[ToolCallIO]:
    if tool_name not in registry:
        return None

    tool_def = registry[tool_name]
    input_params = {}
    format_args = {}

    # Generate random-but-valid parameters
    for param_name, param_def in tool_def["params"].items():
        if param_def.get("enum"):
            value = random.choice(param_def["enum"])
        elif param_def["type"] == "int":
            value = (
                random.randint(2, 10)
                if "age" in param_name
                else random.randint(50, 200)
            )
        else:
            value = "mock_string_value"
        input_params[param_name] = value
        format_args[param_name] = value

    # Add random values for the output template
    format_args["price"] = random.randint(150, 220) * 1000
    format_args["amount"] = random.randint(100, 500)
    format_args["volume"] = random.randint(50, 200)
    format_args["chance"] = random.randint(10, 60)

    output_content = tool_def["mock_output_template"].format(**format_args)

    return ToolCallIO(
        function_tool=tool_name,
        input_params=input_params,
        output_content=[output_content],
        success=True,
        latency_ms=random.randint(200, 1500),
    )
