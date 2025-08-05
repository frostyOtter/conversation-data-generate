from typing import List, Dict, Tuple
from google import genai
from loguru import logger
from pydantic import BaseModel
from src.conversation_models import Scenarios, UserQuery


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

    def _generate_structured_content(
        self, prompt: str, basemodel: BaseModel
    ) -> BaseModel:
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[basemodel],
                ),
            )
            return response.parsed

        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return f"//Error generating content: {e}//"

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
    ) -> BaseModel:
        formatted_history = self._format_history(history)
        if not history:  # Initial query
            prompt = f"""
                These are the list current available tool
                {list_tools_name}
                ---
                You are simulating a user talking to an AI assistant.
                Your persona: '{persona}'.
                The conversation topic: '{topic}'.
                Generate a single, short, initial question a user would ask about this topic.
                Do not add any preamble or explanation. Just simulate a user interacting with agentic chatbot.
                Keep the language practical, simple terms and accessible.
                Based on the user message, inquiries, suggest the approriate actions for the assistsant.
                Suggest tools to support the action by list down the tool name or just leave None if not needed.
                """
        else:  # Follow-up query
            prompt = f"""
                These are the list current available tool
                {list_tools_name}
                ---
                You are simulating a user in an ongoing conversation with an AI assistant.
                Your persona: '{persona}'.

                Here is the conversation history so far:
                --- HISTORY ---
                {formatted_history}
                --- END HISTORY ---

                Based on the assistant's last response and the entire conversation context, generate a single, short, relevant follow-up question.
                Do not add any preamble or explanation. Just simulate a user interacting with agentic chatbot.
                Keep the language practical, simple terms and accessible.
                Based on the user message, inquiries, suggest the approriate actions for the assistsant.
                Suggest tools to support the action by list down the tool name or just leave None if not needed.
                """
        return self._generate_structured_content(prompt, UserQuery)

    def generate_assistant_response(
        self,
        history: List[Dict[str, str]],
        user_persona: str,
        tool_outputs: List[str],
        suggest_actions: List[str],
    ) -> Tuple[str, int, int, int]:
        formatted_history = self._format_history(history)

        prompt = f"""
            You are an AI assistant for a real-time conversation. Your responses must be short, brief, and respectful.

            Your primary goal is to be easily understood by {user_persona}.

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

            1. **Communication Style:**
            - **Use Simple, Practical Language:** Translate complex scientific or technical terms into simple, common language.
            - **Be Direct:** Skip introductory phrases like "I can help with that." Jump straight to the core of the answer or question.
            - **Explain Acronyms:** If the user uses an acronym (e.g., "NPK"), spell it out in your response (e.g., "Nitrogen, Phosphorus, and Potassium").
            - **Be Factual:** Provide unbiased, factual information. Do not offer personal opinions.
            - When user requested to respond in "ภาษาไทย", use polite question particles (e.g., คะ) and statement particles (e.g., ค่ะ).

            2. **Response Strategy**
            Your response strategy depends on the clarity of the user's message:

            - Reference previous messages before asking for clarification.

            - **If the provided question is specific and clear (e.g., "How long does X last?"):**
                1. Leverage available tools to retrieve relevant context
                2. Provide a direct, factual answer using simple language.
                3. After your answer, ask a relevant follow-up question to see if they need more detail.
            - **If the user's inquiry is vague or broad (e.g., "My tree looks sick"):**
                1. Do not guess an answer.
                2. Immediately ask specific clarifying questions to get the details you need.
                3. Suggest to the user that you can diagnose from a picture, one time only.
            - **If the user want to gauge your knowledge (e.g., "Tell me about X", "Describe the image"):**
                1. Leverage available tools to retrieve relevant context
                2. Formulate a comprehensive and satisfactory response.

            3. **Examples**

            - **Example 1:** Responding to a Vague Problem
                - **Input:** "There are strange insects in my orchard."
                - **Output:** "Can you describe the insects? Are they, like, tiny or huge? Any little eggs around, or are the leaves all chewed up?? A pic would be super helpful for confirming what they are. Does that sound good?"

            - **Example 2:** Simplifying a Technical Term
                - **Input:** "I heard my neighbor's trees have Phytophthora. What should I look for?"
                - **Output:** "The main signs of that root and trunk rot fungus are yellowing leaves that drop off, branches dying back, and dark, oozing sap on the trunk near the ground. Wanna know a lil' hack to deal with it? 😉?"

            - **Example 3:** Be useful, expanded tips
                - **Input:** "How do I transform raw durian into higher-value products?"
                - **Output:** "You can process durian into higher-value products like durian paste, frozen durian, durian chips, durian candies, or durian ice cream. These are straight up money printers compared to selling fresh fruit. No cap, your profit margins are about to be bussin! Which one's hitting different for you?"
            """
        return self._generate_content(prompt)

    def generate_scenarios(self, topic: str, num_scenarios: int) -> List[BaseModel]:
        prompt = f"""
        Generate {num_scenarios} diverse, realistic scenarios related to '{topic}'.
        Each scenario should represent a different use case, problem, or situation within this domain.
        
        Format your response as a numbered list with:
        - Scenario name (2-4 words)
        - User persona (brief description)
        - Specific situation or problem they face
        
        Example format:
        1. **Beginner Setup** - New hobbyist who just started and needs basic guidance.
        2. **Troubleshooting Issue** - Experienced user facing a specific technical problem.
        3. **Greeting and General conversation** - This encompasses initial greetings, casual conversation, and exploratory interactions where users seek to assess the AI service's capabilities.
        4. **Incorporate To Answer** - This covers scenarios where users provide information independently, without integrating it into a structured query or prompt.
        
        Make each scenario distinct and realistic for the '{topic}' domain.
        """

        response = self._generate_structured_content(prompt, Scenarios)
        scenario_list = response[0].scenario_list

        if not scenario_list:
            return []

        return (
            scenario_list[:num_scenarios]
            if len(scenario_list) >= num_scenarios
            else scenario_list
        )
