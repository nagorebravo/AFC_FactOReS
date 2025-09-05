import uuid
from typing import Any

import huggingface_hub
from gradio_client import Client

from src.api_calls.api_handler import ApiHandler, get_api_key


class GracioSpace(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = Client(model)
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        **kwargs: Any,
    ) -> Any:
        """
        Call the Hugginface Spaces Gradio API with the given data and return the response.

        Args:
            kwargs: The data to send to the API.
        Returns:
            Any: The response from the API.
        """
        result = self.client.predict(**kwargs)

        self.add_cost(0.0)

        return result


class Flux(GracioSpace):
    def __init__(self):
        super().__init__(model="black-forest-labs/FLUX.1-schnell", pretty_name="flux")

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
        width, height = size.strip().split("x")
        width, height = int(width), int(height)
        path = self.client.predict(image_description, width=width, height=height)[0]
        self.add_cost(0.0)
        try:
            api_key = get_api_key("HF_TOKEN")
        except KeyError:
            api_key = True
        image_name = f"{uuid.uuid4()}.webp"
        _ = huggingface_hub.upload_file(
            repo_id="Iker/FCImages",
            repo_type="dataset",
            token=api_key,
            path_in_repo=image_name,
            path_or_fileobj=path,
        )

        return (
            f"https://huggingface.co/datasets/Iker/FCImages/resolve/main/{image_name}",
            0.0,
        )
