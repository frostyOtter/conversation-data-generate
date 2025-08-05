import os
import uuid
import random
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger

from src.conversation_models import (
    Conversation,
    Turn,
    UserMessage,
    AssistantResponse,
    LatencyStats,
    TokenUsage,
    TurnSummary,
    ConversationSummary,
    UserMetadata,
)
from src.toolsets import (
    generate_mock_tool_call,
    read_tools_registry_from_yaml_file,
)
from src.content_generator import ContentGenerator


def generate_conversation(
    generator: ContentGenerator, topic: str, persona: str, turns: int
) -> Conversation:
    conv_id = f"conv_{uuid.uuid4()}"
    start_time = datetime.now()
    all_turns = []
    total_latency = 0
    last_message_id = None
    conversation_history = []
    logger.info(
        "\n" + "=" * 50 + f"\nGenerating new conversation on '{topic}'...\n" + "=" * 50
    )

    tools_registry, topic_keyword_to_tools = read_tools_registry_from_yaml_file(
        os.path.join("src", "tools_registry", "durian_cultivation.yaml")
    )

    for i in range(turns):
        turn_start_time = start_time + timedelta(minutes=i * 2)
        logger.info(f"Turn {i + 1}: Generating user query...")

        user_responses = generator.generate_user_query(
            conversation_history, topic, persona, list(tools_registry.keys())
        )
        user_query = user_responses[0].user_message
        suggest_actions = user_responses[0].suggest_actions
        suggest_tools = user_responses[0].suggest_tools

        conversation_history.append({"role": "user", "text": user_query})
        user_msg_id = f"user_msg_{uuid.uuid4()}"
        all_turns.append(
            Turn(
                turn_id=(i * 2) + 1,
                initiator_role="user",
                started_at=turn_start_time,
                user_message=UserMessage(
                    message_id=user_msg_id,
                    parent_id=last_message_id,
                    text=user_query,
                    timestamp=turn_start_time,
                ),
            )
        )
        last_message_id = user_msg_id

        # Time break
        time.sleep(0.5)

        logger.info(f"Turn {i + 1}: Generating assistant response...")
        if suggest_tools:
            tool_calls = [
                generate_mock_tool_call(tool, tools_registry) for tool in suggest_tools
            ]
            tool_calls = [tc for tc in tool_calls if tc]  # Filter out None
            tool_outputs = [tc.output_content[0] for tc in tool_calls if tc.success]
        else:
            tool_calls = []
            tool_outputs = ["Not need"]

        (
            assistant_text,
            prompt_token_count,
            completion_token_count,
            total_token_count,
        ) = generator.generate_assistant_response(
            conversation_history, persona, tool_outputs, suggest_actions
        )
        conversation_history.append({"role": "assistant", "text": assistant_text})
        asst_msg_id = f"asst_msg_{uuid.uuid4()}"
        latency = sum(tc.latency_ms for tc in tool_calls) + random.randint(500, 1500)
        total_latency += latency
        all_turns.append(
            Turn(
                turn_id=(i * 2) + 2,
                initiator_role="assistant",
                started_at=turn_start_time,
                assistant_response=AssistantResponse(
                    message_id=asst_msg_id,
                    parent_id=last_message_id,
                    text=assistant_text,
                    tool_calls=tool_calls,
                    assistant_success=True,
                    function_call_success=True,
                    final_output_success=True,
                    latency=LatencyStats(total_ms=latency, inference_ms=latency - 200),
                    token_usage=TokenUsage(
                        prompt_tokens=prompt_token_count,
                        completion_tokens=completion_token_count,
                        total_tokens=total_token_count,
                    ),
                    generated_at=turn_start_time + timedelta(seconds=5),
                    received_at=turn_start_time + timedelta(seconds=1),
                ),
                summary=TurnSummary(intent=f"intent_{topic}", tools_used=suggest_tools),
            )
        )
        last_message_id = asst_msg_id

    return Conversation(
        id=conv_id,
        language="en",
        status="completed",
        turns=all_turns,
        summary=ConversationSummary(
            total_turns=len(all_turns),
            average_processing_time_ms=int(total_latency / turns) if turns > 0 else 0,
            average_latency_ms=int(total_latency / turns) if turns > 0 else 0,
        ),
        tags=[topic, persona],
        user_metadata=UserMetadata(user_id=f"user_{random.randint(1000, 9999)}"),
    )


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info("❌ ERROR: GEMINI_API_KEY environment variable not found.")
        logger.info(
            "Please create a .env file and add your key, or set the environment variable."
        )
    else:
        logger.info("✅ Gemini API Key loaded successfully.")
        logger.info("🤖 Conversation Data Generator (With History Aware)")
        logger.info("-" * 60)

        main_topic = input(
            "Enter the main topic (e.g., durian cultivation, weather forecast: "
        )
        persona = input(
            "Enter a user persona (e.g., durian farmer, tourist in Bangkok): "
        )

        try:
            num_conversations = int(
                input("Enter the number of conversations to generate: ")
            )
            num_turns = int(
                input("Enter the number of user-assistant turns per conversation: ")
            )
        except ValueError:
            logger.info("Invalid input. Please enter whole numbers.")
            exit()

        logger.info("-" * 60)

        # Content Generator
        content_generator = ContentGenerator(api_key, "Gemini")

        # Datetime
        today_datetime = datetime.today()
        today_datetime = today_datetime.strftime("%d %B")

        # Save path
        save_path = "conversations"
        if not os.path.exists(save_path):
            logger.warning(
                f"Save path: {save_path} not found, attemp to create folder."
            )
            os.mkdir(save_path)

        all_files = len(
            [file for file in os.listdir(save_path) if file.endswith(".json")]
        )

        # Generate scenarios first
        logger.info(
            f"🎯 Generating {num_conversations} scenarios for '{main_topic}'..."
        )
        scenarios = content_generator.generate_scenarios(main_topic, num_conversations)
        if len(scenarios) == 0:
            logger.error(f"Total scenario generated: {len(scenarios)}")
        else:
            for i, scenario in enumerate(scenarios):
                generated_conv = generate_conversation(
                    content_generator,
                    scenario.situation,
                    scenario.user_persona,
                    num_turns,
                )
                filename = os.path.join(
                    save_path,
                    f"conversation_{all_files + i + 1}.json",
                )

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(generated_conv.model_dump_json(indent=2))

                logger.info(
                    f"\n✅ Conversation {all_files + i + 1} successfully generated and saved to '{filename}'"
                )

            logger.info("\n🎉 All conversations generated!")
