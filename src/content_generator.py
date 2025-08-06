import json
from typing import List, Dict, Tuple, Literal
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.settings import ModelSettings
from src.conversation_models import (
    Scenarios,
    UserQuery,
    ToolCallIO,
)


class ContentGenerator:
    """Generates conversational content using the Gemini API with history."""

    def __init__(self, api_key: str, provider: str = "Gemini"):
        if provider == "Gemini":
            self.client = self._initialize_gemini_service(api_key=api_key)
        else:
            raise ValueError(f"Currently not supported provider: {provider}")

    def _initialize_gemini_service(self, api_key: str) -> PydanticAgent:
        from pydantic_ai.models.gemini import GeminiModel
        from pydantic_ai.providers.google_gla import GoogleGLAProvider

        model = GeminiModel(
            model_name="gemini-2.0-flash", provider=GoogleGLAProvider(api_key=api_key)
        )
        return PydanticAgent(model=model)

    def _generate_content(self, prompt: str) -> Tuple[str, int, int, int]:
        try:
            response = self.client.run_sync(
                user_prompt=prompt,
                output_type=str,
                model_settings=ModelSettings(temperature=0.8, top_p=0.95),
            )

            # Retrieve token usages
            token_usage = response.usage()
            prompt_token_count = token_usage.request_tokens
            completion_token_count = token_usage.response_tokens
            total_token_count = token_usage.total_tokens

            # Retrieve response text
            responses = response.output

            return (
                responses,
                prompt_token_count,
                completion_token_count,
                total_token_count,
            )

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"// Error generating content: {e} //", 0, 0, 0

    def _generate_structured_content(
        self, prompt: str, basemodel: BaseModel
    ) -> BaseModel | None:
        try:
            response = self.client.run_sync(
                user_prompt=prompt,
                output_type=basemodel,
                model_settings=ModelSettings(temperature=0.9, top_p=0.95),
            )
            return response.output

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return None

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Formats the history list into a readable string for the prompt."""
        if not history:
            return "This is the beginning of the conversation."
        return "\n".join(
            [f"{item['role'].capitalize()}: {item['text']}" for item in history]
        )

    def generate_user_query(
        self,
        history: List[Dict[str, str]],
        topic: str,
        persona: str,
        list_tools_name: List[str],
        language: Literal["English", "ภาษาไทย"] = "English",
    ) -> BaseModel:
        formatted_history = self._format_history(history)
        if not history:  # Initial query
            prompt = f"""
--- AVAILABLE TOOLS ---
These are the list current available tool
{json.dumps(list_tools_name, indent=4, ensure_ascii=False)}
--- END AVAILABLE TOOLS ---

--- INSTRUCTIONS ---
You are simulating a user talking to an AI assistant.
Your persona: '{persona}'.
The conversation topic: '{topic}'.
Language: {language}

Generate a single, short, initial question a user would ask about this topic.
Do not add any preamble or explanation. Just simulate a user interacting with agentic chatbot.
Keep the language practical, simple terms and accessible.
Based on the user message, inquiries, suggest the approriate actions for the assistsant.
Suggest tools that best support the inquiry by list down the tool name or just leave None if not needed.
--- END INSTRUCTIONS ---
"""
        else:  # Follow-up query
            prompt = f"""
--- AVAILABLE TOOLS ---
These are the list current available tool
{json.dumps(list_tools_name, indent=4, ensure_ascii=False)}
--- END AVAILABLE TOOLS ---

--- INSTRUCTIONS ---
You are simulating a user in an ongoing conversation with an AI assistant.
Your persona: '{persona}'.
Language: {language}

Here is the conversation history so far:
--- HISTORY ---
{formatted_history}
--- END HISTORY ---

Based on the assistant's last response and the entire conversation context, generate a single, short, relevant follow-up question.
Do not add any preamble or explanation. Just simulate a user interacting with agentic chatbot.
Keep the language practical, simple terms and accessible.
Based on the user message, inquiries, suggest the approriate actions for the assistsant.
Based on tool's description, suggest tools that best support the inquiry by list down the tool name or just leave None if not needed.
--- END INSTRUCTIONS ---
"""
        return self._generate_structured_content(prompt.strip(), UserQuery)

    def generate_assistant_response(
        self,
        history: List[Dict[str, str]],
        tool_outputs: List[str],
        suggest_actions: List[str],
        language: Literal["English", "ภาษาไทย"] = "English",
    ) -> Tuple[str, int, int, int]:
        formatted_history = self._format_history(history)

        prompt = f"""
You are a friendly, laid-back, and knowledgeable agricultural expert. Your goal is to make complex topics easy and fun to understand.
Your tone is formal, show respecting, encouraging, and approachable.

Here is the conversation history so far. The last message is the user's current query.
--- HISTORY ---
{formatted_history}
--- END HISTORY ---

--- SUGGEST ACTIONS ---
{suggest_actions}
--- END SUGGEST ACTIONS ---

You have just used your internal tools to gather information for your response and received the following data:
--- TOOL OUTPUTS ---
{tool_outputs}
--- END TOOL OUTPUTS ---

--- COMMUNICATION STYLE ---
- **Use Simple, Practical Language:** Translate complex scientific or technical terms into simple, common language.
- **Be Direct:** Skip introductory phrases like "I can help with that." Jump straight to the core of the answer or question. Don't repeat the question.
- **Explain Acronyms:** If the user uses an acronym (e.g., "NPK"), spell it out in your response (e.g., "Nitrogen, Phosphorus, and Potassium").
- **Be Factual but Friendly:** Provide factual information, but frame it with your encouraging and informal persona. Frame tips and suggestions as exciting "hacks" or inside knowledge (e.g., "Wanna know a lil' hack to deal with it? 😉?").
- **Language:** {language}
--- END COMMUNICATION STYLE ---

--- RESPOND STRATEGY ---
Your response strategy depends on the clarity of the user's message:

- Reference previous messages before asking for clarification.
- **If the provided question is specific and clear (e.g., "How long does X last?"):**
    1. Leverage available tools to retrieve relevant context
    2. Provide a direct, factual answer **in your informal persona and simple language.**
    3. After your answer, ask a **conversational follow-up question** to keep the chat going (e.g., "Which one's hitting different for you?", "Does that sound good?").
- **If the user's inquiry is vague or broad (e.g., "My tree looks sick"):**
    1. Do not guess an answer.
    2. Immediately ask specific clarifying questions to get the details you need.
    3. Suggest to the user that you can diagnose from a picture.
- **If the user want to gauge your knowledge (e.g., "Tell me about X", "Describe the image"):**
    1. Leverage available tools to retrieve relevant context
    2. Formulate a comprehensive and satisfactory response.
--- END RESPOND STRATEGY ---

--- EXAMPLES ---
- **Example 1:** Responding to a Vague Problem
    - **Input:** "There are strange insects in my orchard."
    - **Output:** "Can you describe the insects? Are they, like, tiny or huge? Any little eggs around, or are the leaves all chewed up?? A pic would be super helpful for confirming what they are. Does that sound good?"

- **Example 2:** Simplifying a Technical Term
    - **Input:** "I heard my neighbor's trees have Phytophthora. What should I look for?"
    - **Output:** "The main signs of that root and trunk rot fungus are yellowing leaves that drop off, branches dying back, and dark, oozing sap on the trunk near the ground. Wanna know a lil' hack to deal with it? 😉?"

- **Example 3:** Be useful, expanded tips
    - **Input:** "How do I transform raw durian into higher-value products?"
    - **Output:** "You can process durian into higher-value products like durian paste, frozen durian, durian chips, durian candies, or durian ice cream. These are straight up money printers compared to selling fresh fruit. No cap, your profit margins are about to be bussin! Which one's hitting different for you?"
--- END EXAMPLES ---
"""
        return self._generate_content(prompt.strip())

    def generate_scenarios(self, topic: str, num_scenarios: int) -> List[BaseModel]:
        prompt = f"""
Generate {num_scenarios} diverse, realistic scenarios related to '{topic}'.
Each scenario should represent a different use case, problem, or situation within this domain.

Format your response as a numbered list with:
- Scenario name (2-3 sentences)
- User persona (short and brief description)
- Specific situation or problem they face (2-3 sentences)

Example format:
1. **Beginner Setup** - New hobbyist who just started and needs basic guidance.
2. **Troubleshooting Issue** - Experienced user facing a specific technical problem.
3. **Greeting and General conversation** - This encompasses initial greetings, casual conversation, and exploratory interactions where users seek to assess the AI service's capabilities.
4. **Incorporate To Answer** - This covers scenarios where users provide information independently, without integrating it into a structured query or prompt.

Make each scenario distinct and realistic for the '{topic}' domain.
"""

        response = self._generate_structured_content(prompt, Scenarios)
        scenario_list = response.scenario_list

        if not scenario_list:
            return []

        return (
            scenario_list[:num_scenarios]
            if len(scenario_list) >= num_scenarios
            else scenario_list
        )

    def generate_mock_tool_call(
        self, history: List[Dict[str, str]], tool_name: str, tool_registry: Dict
    ) -> BaseModel | None:
        tool_def = tool_registry.get(tool_name, "")
        if not tool_def:
            logger.warning(f"Not supported tool name: {tool_name}")
            return None
        formatted_history = self._format_history(history)

        prompt = f"""
--- HISTORY ---
{formatted_history}
--- END HISTORY ---

--- TOOL ---
Tool name: {tool_name}
Tool definition: {json.dumps(tool_def, indent=4, ensure_ascii=False)}
--- END TOOL ---

Act as a tool, select the approriate params and help me generate the result from the tool based on the input.
"""

        responses = self._generate_structured_content(prompt.strip(), ToolCallIO)
        return responses
