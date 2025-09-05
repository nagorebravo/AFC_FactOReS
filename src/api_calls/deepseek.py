from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import BaseModel

from src.api_calls.api_handler import ApiHandler, get_api_key

_PRICING = {
    "deepseek-chat_input": 0.14 / 1000000,
    "deepseek-chat_output": 0.28 / 1000000,
}


class GPT(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = OpenAI(
            api_key=get_api_key("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        conversation: List[Dict[str, str]],
        temperature: Optional[int] = None,
        max_tokens: Optional[int] = None,
        pydantic_model: Optional[BaseModel] = None,
    ) -> Tuple[str, float]:
        """
        Call the DEEPSEEK API with the given conversation and return the response.

        Args:
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            temperature (Optional[int]): The temperature to use.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """

        if pydantic_model is None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

        cost = (
            response.usage.prompt_tokens * _PRICING[f"{self.model}_input"]
            + response.usage.completion_tokens * _PRICING[f"{self.model}_output"]
        )

        self.add_cost(cost)

        return response.choices[0].message.content, cost


class DeepseekChat(GPT):
    def __init__(self):
        super().__init__("deepseek-chat", pretty_name="deepseek")
