from functools import lru_cache
from typing import Any

from replicate import Client

from src.api_calls.api_handler import ApiHandler, get_api_key

_PRICING = {
    "black-forest-labs/flux-dev": 0.030,
    "black-forest-labs/flux-pro": 0.055,
    "black-forest-labs/flux-schnell": 0.003,
}

AVAILABLE_ASPECT_RATIOS = [
    "1:1",
    "16:9",
    "21:9",
    "2:3",
    "3:2",
    "4:5",
    "5:4",
    "9:16",
    "9:21",
]


class ReplicateAPI(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = Client(api_token=get_api_key("REPLICATE_API_TOKEN"))
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        **kwargs: Any,
    ) -> Any:
        """
        Call the Replicate API with the given data and return the response.

        Args:
            kwargs: The data to send to the API.
        Returns:
            Any: The response from the API.
        """
        result = self.client.run(**kwargs)

        return result


class Flux(ReplicateAPI):
    def __init__(self):
        super().__init__(model="black-forest-labs/flux-schnell", pretty_name="flux")

    @lru_cache(maxsize=128)
    def get_closest_aspect_ratio(self, width: int, height: int) -> str:
        target_ratio = width / height
        return min(
            AVAILABLE_ASPECT_RATIOS,
            key=lambda x: abs(eval(x.replace(":", "/")) - target_ratio),
        )

    def __call__(self, image_description: str, size: str, **kwargs) -> str:
        """
        Call the FLUX API with the given prompt and size and return the response.

        Args:
            image_description: The prompt to send to the API.
            size: The size of the image to generate.
            kwargs: Unused. Here for compatibility with other APIs.

        Returns:
            str: The URL of the generated image.
        """
        width, height = map(int, size.strip().split("x"))
        aspect_ratio = self.get_closest_aspect_ratio(width, height)

        result = self.client.run(
            self.model,
            input={
                "prompt": image_description,
                "num_outputs": 1,
                "aspect_ratio": aspect_ratio,
                "output_format": "webp",
                "output_quality": 80,
            },
            use_file_output=False,
        )[0]

        cost = _PRICING[self.model]
        self.add_cost(cost)

        return result, cost
