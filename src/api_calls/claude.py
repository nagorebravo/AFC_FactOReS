import base64
from typing import Dict, List, Optional, Tuple

import httpx
from anthropic import Anthropic

from src.api_calls.api_handler import ApiHandler, get_api_key

_PRICING = {
    "haiku_input": 0.25 / 1000000,
    "haiku_output": 1.25 / 1000000,
    "sonnet_input": 3 / 1000000,
    "sonnet_output": 15 / 1000000,
    "opus_input": 15 / 1000000,
    "opus_output": 75 / 1000000,
}


class Claude(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = Anthropic(
            api_key=get_api_key("ANTHROPIC_API_KEY"),
        )

        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        conversation: List[Dict[str, str]],
        max_tokens: int = 32000,
        temperature: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, float]:
        """
        Call the Claude API with the given conversation and return the response.

        Args:
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[int]): The temperature to use.

        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """

        kwargs = {}
        if conversation[0]["role"] == "system":
            system_prompt = conversation[0]["content"]
            conversation = conversation[1:]
            kwargs["system"] = system_prompt

        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=conversation,
            **kwargs,
        )

        cost = (
            response.usage.input_tokens * _PRICING[f"{self.api_name}_input"]
            + response.usage.output_tokens * _PRICING[f"{self.api_name}_output"]
        )

        self.add_cost(cost)

        return response.content[0].text, cost


class Haiku(Claude):
    def __init__(self):
        super().__init__("claude-3-haiku-20240307", pretty_name="haiku")


class Sonnet(Claude):
    def __init__(self):
        super().__init__("claude-3-sonnet-20240229", pretty_name="sonnet")


class Opus(Claude):
    def __init__(self):
        super().__init__("claude-3-opus-20240229", pretty_name="opus")


class ClaudeImageDescription(Claude):
    def __call__(
        self,
        image_url: str,
        max_tokens: Optional[int] = None,
        prompt: Optional[
            str
        ] = "Describe la imagen lo mejor posible. Describe todos los elementos que aparecen.",
    ) -> Tuple[str, float]:
        """
        Describe the image using the OpenAI chat model.

        Args:
            image_url (str): The url of the image to describe.
            max_tokens (int): The maximum number of tokens to generate.
            prompt (str): The prompt to use for the description.

        Returns:
            Tuple[str, float]: The description of the image and the cost of the call.
        """

        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        image_type = image_url.split(".")[-1]

        if image_type == "jpg":
            image_type = "jpeg"

        if image_type not in ["jpeg", "png", "gif", "webp"]:
            raise ValueError(
                "Image type not supported. Please use a JPEG, PNG, GIF or WEBP image."
            )
        print(f"image/{image_type}")
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": f"image/{image_type}",
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return super().__call__(conversation, max_tokens=max_tokens)


class HaikuImageDescription(ClaudeImageDescription):
    def __init__(self):
        super().__init__("claude-3-haiku-20240307", pretty_name="haiku")


class SonnetImageDescription(ClaudeImageDescription):
    def __init__(self):
        super().__init__("claude-3-5-sonnet-20241022", pretty_name="sonnet")


class OpusImageDescription(ClaudeImageDescription):
    def __init__(self):
        super().__init__("claude-3-opus-20240229", pretty_name="opus")
