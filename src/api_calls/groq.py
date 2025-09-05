from typing import Dict, List, Optional, Tuple

from groq import Groq

from src.api_calls.api_handler import ApiHandler, get_api_key

_PRICING = {
    "llama-3.1-8b-instant_input": 0.00 / 1000000,
    "llama-3.1-8b-instant_output": 0.00 / 1000000,
    "llama-3.1-70b-versatile_input": 0.00 / 1000000,
    "llama-3.1-70b-versatile_output": 0.00 / 1000000,
    "llama-3.1-405b-reasoning_input": 0.00 / 1000000,
    "llama-3.1-405b-reasoning_output": 0.00 / 1000000,
}


class GROQ(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = Groq(api_key=get_api_key("GROQ_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        conversation: List[Dict[str, str]],
        temperature: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, float]:
        """
        Call the OpenAI API with the given conversation and return the response.

        Args:
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            temperature (Optional[int]): The temperature to use.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        cost = (
            response.usage.prompt_tokens * _PRICING[f"{self.model}_input"]
            + response.usage.completion_tokens * _PRICING[f"{self.model}_output"]
        )

        self.add_cost(cost)

        return response.choices[0].message.content, cost


class Llama38B(GROQ):
    def __init__(self):
        super().__init__("llama-3.1-8b-instant", pretty_name="llama3-8b")


class Llama370B(GROQ):
    def __init__(self):
        super().__init__("llama-3.1-70b-versatile", pretty_name="llama3-70b")


class Llama405B(GROQ):
    def __init__(self):
        super().__init__("llama-3.1-405b-reasoning", pretty_name="llama3-405b")
