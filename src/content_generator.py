from typing import List, Dict, Tuple
from google import genai
from loguru import logger


class ContentGenerator:
    """Generates conversational content using the Gemini API with history."""

    def __init__(self, api_key: str, provider: str = "Gemini"):
        if provider == "Gemini":
            self.client = self._initialize_gemini_service(api_key=api_key)
        else:
            raise ValueError(f"Currently not supported provider: {provider}")

    def _initialize_gemini_service(self, api_key: str) -> genai.Client:
        return genai.Client(api_key=api_key)

    def _generate_content(self, prompt: str) -> Tuple[str, int, int, int]:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash", contents=prompt
            )

            # Retrieve token usages
            prompt_token_count = response.usage_metadata.prompt_token_count
            completion_token_count = response.usage_metadata.candidates_token_count
            total_token_count = response.usage_metadata.total_token_count

            # Retrieve response text
            responses = response.text

            return (
                responses,
                prompt_token_count,
                completion_token_count,
                total_token_count,
            )

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"// Error generating content: {e} //", 0, 0, 0

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Formats the history list into a readable string for the prompt."""
        if not history:
            return "This is the beginning of the conversation."
        return "\n".join(
            [f"{item['role'].capitalize()}: {item['text']}" for item in history]
        )

    def generate_user_query(
        self, history: List[Dict[str, str]], topic: str, persona: str
    ) -> Tuple[str, int, int, int]:
        formatted_history = self._format_history(history)
        if not history:  # Initial query
            prompt = f"""
                You are simulating a user talking to an AI assistant.
                Your persona: '{persona}'.
                The conversation topic: '{topic}'.
                Generate a single, short, initial question a user would ask about this topic.
                Do not add any preamble or explanation. Just provide the question.
                """
        else:  # Follow-up query
            prompt = f"""
                You are simulating a user in an ongoing conversation with an AI assistant.
                Your persona: '{persona}'.

                Here is the conversation history so far:
                --- HISTORY ---
                {formatted_history}
                --- END HISTORY ---

                Based on the assistant's last response and the entire conversation context, generate a single, short, relevant follow-up question.
                Do not add any preamble or explanation. Just provide the next question from the user's perspective.
                """
        return self._generate_content(prompt)

    def generate_assistant_response(
        self, history: List[Dict[str, str]], tool_outputs: List[str]
    ) -> Tuple[str, int, int, int]:
        formatted_history = self._format_history(history)
        prompt = f"""
            You are an advanced, helpful AI assistant.

            Here is the conversation history so far. The last message is the user's current query.
            --- HISTORY ---
            {formatted_history}
            --- END HISTORY ---

            You have just used your internal tools to gather information for your response and received the following data:
            --- TOOL OUTPUTS ---
            {tool_outputs}
            --- END TOOL OUTPUTS ---

            Synthesize the information from your tools into a concise, conversational, and helpful response.
            Your response must directly address the user's last message, continuing the conversation naturally.
            Do not mention your tools explicitly (e.g., don't say "my tool says...").
            """
        return self._generate_content(prompt)
