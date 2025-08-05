import json
import os
import sys
from typing import Dict, Any, List


def create_markdown_table(data: Dict[str, Any], headers: List[str]) -> str:
    """Creates a two-column markdown table from a dictionary."""
    if not data:
        return ""

    # Create the header and separator lines
    header = f"| {headers[0]} | {headers[1]} |\n"
    separator = "| :--- | :--- |\n"

    # Create the body of the table
    body = ""
    for key, value in data.items():
        # Ensure values are strings and sanitize pipes to prevent breaking the table
        sanitized_value = str(value).replace("|", "\\|") if value is not None else "N/A"
        body += f"| **{key}** | {sanitized_value} |\n"

    return header + separator + body


def format_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    """Formats the tool_calls list into a detailed Markdown section."""
    if not tool_calls:
        return ""

    parts = ["#### 🔧 Tool Calls"]
    for i, call in enumerate(tool_calls):
        success_icon = "✅" if call.get("success") else "❌"
        latency = call.get("latency_ms", "N/A")

        # More compact and informative header for each tool call
        parts.append(
            f"**{i + 1}. Function:** `{call.get('function_tool', 'N/A')}` ({success_icon} | ⏱️ {latency} ms)"
        )

        input_params = call.get("input_params", {})
        if input_params:
            parts.append("**Input:**")
            parts.append(f"```json\n{json.dumps(input_params, indent=2)}\n```")

        output_content = call.get("output_content", [])
        if output_content:
            parts.append("**Output:**")
            output_str = "\n".join(map(str, output_content))
            parts.append(f"```text\n{output_str}\n```")

        if "error" in call and call["error"]:
            error_msg = call["error"].get("message", "No error message.")
            parts.append(f"**Error:** 🚨 `{error_msg}`")

        parts.append("\n")

    return "\n".join(parts)


def format_turn(turn: Dict[str, Any]) -> str:
    """Formats a single turn of a conversation."""
    turn_id = turn.get("turn_id", "N/A")
    parts = []

    if "user_message" in turn and turn["user_message"]:
        msg = turn["user_message"]
        parts.append(f"### 💬 Turn {turn_id}: User")
        parts.append(f"> {msg.get('text', '')}\n")

        if "attachments" in msg and msg["attachments"]:
            parts.append("**Attachments:**")
            for att in msg["attachments"]:
                parts.append(
                    f"- [{att.get('attachment_type', 'file')}]({att.get('url')})"
                )

        parts.append(f"Timestamp: `{msg.get('timestamp')}`")

    elif "assistant_response" in turn and turn["assistant_response"]:
        resp = turn["assistant_response"]
        success_icon = "✅" if resp.get("assistant_success") else "❌"
        parts.append(f"### 🤖 Turn {turn_id}: Assistant {success_icon}")
        parts.append(f"{resp.get('text', '')}\n")

        parts.append(format_tool_calls(resp.get("tool_calls", [])))

        # --- Performance & Stats Table ---
        stats_data = {}
        latency = resp.get("latency", {})
        if latency:
            stats_data["⏱️ Latency"] = (
                f"{latency.get('total_ms')} ms (Network: {latency.get('network_ms', 'N/A')} ms, Inference: {latency.get('inference_ms', 'N/A')} ms)"
            )

        token_usage = resp.get("token_usage", {})
        if token_usage:
            stats_data["🎟️ Token Usage"] = (
                f"**{token_usage.get('total_tokens')} Total** (Prompt: {token_usage.get('prompt_tokens', 0)}, Completion: {token_usage.get('completion_tokens', 0)})"
            )

        feedback = resp.get("feedback", {})
        if feedback and feedback.get("thumbs_up") is not None:
            feedback_icon = "👍" if feedback["thumbs_up"] else "👎"
            comment = (
                f" *{feedback.get('comment', '')}*" if feedback.get("comment") else ""
            )
            stats_data["Feedback"] = f"{feedback_icon}{comment}"

        if stats_data:
            parts.append("#### Performance & Stats")
            parts.append(
                create_markdown_table(stats_data, headers=["Metric", "Details"])
            )

        if "error" in resp and resp["error"]:
            error_msg = resp["error"].get("message", "No error message.")
            parts.append(f"\n**Error:** 🚨 `{error_msg}`")

    return "\n".join(parts) + "\n---"


def convert_json_to_markdown(conversation_data: Dict[str, Any]) -> str:
    """Converts a conversation JSON object to a Markdown string."""
    parts = [f"# 📊 Conversation Report: `{conversation_data.get('id')}`", "---"]

    # --- Overview Section ---
    parts.append("## ⚙️ Overview")
    summary = conversation_data.get("summary", {})
    user_meta = conversation_data.get("user_metadata", {})

    overview_data = {
        "Status": f"**{conversation_data.get('status', 'N/A').title()}**",
        "Total Turns": summary.get("total_turns", "N/A"),
        "Avg. Latency": f"{summary.get('average_latency_ms', 'N/A')} ms",
        "Language": conversation_data.get("language", "N/A"),
        "User ID": f"`{user_meta.get('user_id', 'N/A')}`",
        "Tags": (
            ", ".join(f"`{tag}`" for tag in conversation_data.get("tags", []))
            if conversation_data.get("tags")
            else "None"
        ),
    }
    parts.append(create_markdown_table(overview_data, headers=["Metric", "Value"]))
    parts.append("---\n")

    # --- Turns Section ---
    parts.append("## Conversation Turns")
    for turn in conversation_data.get("turns", []):
        parts.append(format_turn(turn))

    return "\n".join(parts)


# Main execution block
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python converter.py <script_number>")
        sys.exit(1)

    script_number = sys.argv[1]

    conversation_path = "conversations"
    script_name = f"conversation_{script_number}"
    output_path = "conversations_md"

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not os.path.exists(conversation_path):
        os.makedirs(conversation_path)

    json_file_path = os.path.join(conversation_path, script_name + ".json")
    markdown_file_path = os.path.join(output_path, script_name + ".md")

    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        markdown_output = convert_json_to_markdown(data)

        with open(markdown_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)

        print(f"✅ Successfully converted '{json_file_path}' to '{markdown_file_path}'")

    except FileNotFoundError:
        print(f"❌ Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"❌ Error: The file '{json_file_path}' contains invalid JSON.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
