import json
from typing import Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.api_calls.api_handler import ApiHandler, get_api_key

_PRICING = {
    "gemini-2.0-flash-exp_input": 0.0 / 1000000,
    "gemini-2.0-flash-exp_output": 0.0 / 1000000,
    "gemini-2.0-flash-thinking-exp-1219_input": 0.0 / 1000000,
    "gemini-2.0-flash-thinking-exp-1219_output": 0.0 / 1000000,
    "gemini-1.5-flash_input": 0.075 / 1000000,
    "gemini-1.5-flash_output": 0.30 / 1000000,
    "gemini-1.5-pro_input": 1.25 / 1000000,
    "gemini-1.5-pro_output": 5.0 / 1000000,
}

_roles_dict = {
    "user": "user",
    "assistant": "model",
    "system": "system",
}


class Gemini(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = genai.Client(api_key=get_api_key("GEMINI_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        conversation: List[Dict[str, str]],
        temperature: Optional[int] = None,
        max_tokens: Optional[int] = None,
        pydantic_model: Optional[BaseModel] = None,
        web_search: Optional[bool] = False,
    ) -> Tuple[str, float]:
        """
        Call the OpenAI API with the given conversation and return the response.

        Args:
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            temperature (Optional[int]): The temperature to use.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            pydantic_model (Optional[BaseModel]): The Pydantic model to use for the response.
            web_search (Optional[bool]): Whether to use the web search model.
        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """

        # Get the system prompt if it exists
        if conversation[0]["role"] == "system":
            system_prompt = conversation[0]["content"]
            conversation = conversation[1:]
        else:
            system_prompt = None

        if len(conversation) > 1:
            raise ValueError(
                "The Google API does not support multi-turn conversations."
            )

        user_message = conversation[0]["content"]

        if not web_search:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_schema=pydantic_model,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    response_mime_type="application/json"
                    if pydantic_model is not None
                    else "text/plain",
                ),
            )
        else:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())

            response = self.client.models.generate_content(
                model=self.model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                    response_modalities=["TEXT"],
                    system_instruction=system_prompt,
                    response_schema=pydantic_model,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

        cost = (
            response.usage_metadata.prompt_token_count * _PRICING[f"{self.model}_input"]
            + response.usage_metadata.candidates_token_count
            * _PRICING[f"{self.model}_output"]
        )

        if "thinking" in self.model:
            text = response.candidates[0].content.parts[-1].text
        else:
            text = response.text
            if pydantic_model is not None and not isinstance(text, dict):
                text = json.loads(text)

        return text, cost


class GeminiFlash(Gemini):
    def __init__(self):
        super().__init__("gemini-2.0-flash-exp", pretty_name="gemini-flash")


class GeminiFlashThinking(Gemini):
    def __init__(self):
        super().__init__(
            "gemini-2.0-flash-thinking-exp-1219", pretty_name="gemini-flash-thinking"
        )


class GeminiPro(Gemini):
    def __init__(self):
        super().__init__("gemini-1.5-pro", pretty_name="gemini-pro")
