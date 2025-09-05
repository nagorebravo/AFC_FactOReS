import concurrent.futures
import logging
import os
from typing import Dict, List, Literal, Optional, Tuple

import torch
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from src.api_calls.api_handler import ApiHandler, get_api_key
from src.api_calls.utils import batch

_PRICING = {
    "text-embedding-3-small": 0.02 / 1000000,
    "text-embedding-3-large": 0.13 / 1000000,
    "text-embedding-ada-002": 0.10 / 1000000,
    "gpt-4o_input": 2.5 / 1000000,
    "gpt-4o_output": 10 / 1000000,
    "gpt-4o-2024-08-06_input": 2.5 / 1000000,
    "gpt-4o-2024-08-06_output": 10 / 1000000,
    "gpt-4-turbo_input": 10 / 1000000,
    "gpt-4-turbo_output": 30 / 1000000,
    "gpt-4o-mini_input": 0.150 / 1000000,
    "gpt-4o-mini_output": 0.600 / 1000000,
    "gpt-3.5-turbo-0125_input": 0.50 / 1000000,
    "gpt-3.5-turbo-0125_output": 1.50 / 1000000,
    "gpt-3.5-turbo_finetuned_input": 0.0030 / 1000,
    "gpt-3.5-turbo_finetuned_output": 0.0060 / 1000,
    "dall-e-3_standard_1024x1024": 0.040,
    "dall-e-3_standard_1792x1024": 0.080,
    "dall-e-3_standard_1024x1792": 0.080,
    "dall-e-3_hd_1024x1024": 0.080,
    "dall-e-3_hd_1792x1024": 0.120,
    "dall-e-3_hd_1024x1792": 0.120,
}


class GPT(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        conversation: List[Dict[str, str]],
        temperature: Optional[int] = None,
        max_tokens: Optional[int] = None,
        pydantic_model: Optional[BaseModel] = None,
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

        if pydantic_model is None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=pydantic_model,
            )

        cost = (
            response.usage.prompt_tokens * _PRICING[f"{self.model}_input"]
            + response.usage.completion_tokens * _PRICING[f"{self.model}_output"]
        )

        self.add_cost(cost)

        return (response.choices[0].message.content) if pydantic_model is None else (
            response.choices[0].message.parsed.__dict__
        ), cost


class GPT3(GPT):
    def __init__(self):
        super().__init__("gpt-3.5-turbo-0125", pretty_name="gpt3")


class GPT4(GPT):
    def __init__(self):
        super().__init__("gpt-4-turbo", pretty_name="gpt4")


class GPT4O(GPT):
    def __init__(self):
        super().__init__("gpt-4o-2024-08-06", pretty_name="gpt4o")


class GPT4omini(GPT):
    def __init__(self):
        super().__init__("gpt-4o-mini", pretty_name="gpt4omini")


class GPT4Vision(GPT):
    def __init__(self):
        super().__init__("gpt-4-turbo", pretty_name="gpt4")


class GPT4ImageDescription(GPT4O):
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
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            }
        ]

        return super().__call__(conversation, max_tokens=max_tokens)


class Dalle3(ApiHandler):
    def __init__(self):
        self.model = "dall-e-3"
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        super().__init__("dalle3")

    def __call__(
        self,
        image_description: str,
        size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1792x1024",
        quality: Literal["standard", "hd"] = "standard",
    ) -> Tuple[str, float]:
        """
        Generate images from the given descriptions using the OpenAI chat model.

        Args:
            image_description (str): The description of the image to generate.
            size (str): The size of the images to generate.
            quality (str): The quality of the images to generate.

        Returns:
            Tuple[str, float]: The url of the generated image and the cost of the call.
        """
        response = self.client.images.generate(
            model=self.model,
            prompt=image_description,
            size=size,
            quality=quality,
            n=1,
        )

        cost = _PRICING[f"{self.model}_{quality}_{size}"]
        self.add_cost(cost)

        return response.data[0].url, cost


'''
class Embeddings(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def process_embeddings(self, texts: List[str], batch_size: int, desc: str):
        embeddings = []
        cost = 0.0
        batches = list(batch(texts, batch_size))
        for b in tqdm(batches, desc=desc, disable=len(batches) == 1):
            try:
                response = self.client.embeddings.create(input=b, model=self.model)
                cost += response.usage.total_tokens * _PRICING[self.model]
                embeddings.extend([x.embedding for x in response.data])
            except Exception:
                logging.warning(
                    f"Error processing jaja{desc} jaja with batch size {batch_size}. Batch is {b}"
                    "We will try to process them one by one."
                )
                for text in b:
                    try:
                        response = self.client.embeddings.create(
                            input=[text], model=self.model
                        )
                        cost += response.usage.total_tokens * _PRICING[self.model]
                        embeddings.append(response.data[0].embedding)
                    except Exception as e:
                        raise ValueError(f"Error processing {desc}: {text}\n{e}.")
        return torch.tensor(embeddings), cost
        '''

class Embeddings(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def process_embeddings(self, texts: List[str], batch_size: int, desc: str):
        embeddings = []
        cost = 0.0
        batches = list(batch(texts, batch_size))
        
        for b in tqdm(batches, desc=desc, disable=len(batches) == 1):
            try:
                response = self.client.embeddings.create(input=b, model=self.model)
                cost += response.usage.total_tokens * _PRICING[self.model]
                embeddings.extend([x.embedding for x in response.data])
            except Exception:
                logging.warning(
                    f"Error processing {desc} with batch size {batch_size}. Batch: {b}"
                    " We will try to process them one by one."
                )
                for text in b:
                    try:
                        response = self.client.embeddings.create(
                            input=[text], model=self.model
                        )
                        cost += response.usage.total_tokens * _PRICING[self.model]
                        embeddings.append(response.data[0].embedding)
                    except Exception as e:
                        logging.warning(f"Skipping problematic input: {text}. Error: {e}")
                        continue  # Ignorar el error y continuar con el siguiente texto
        
        return torch.tensor(embeddings), cost

    def __call__(
        self,
        queries: Optional[List[str]] = None,
        passages: Optional[List[str]] = None,
        batch_size: Optional[int] = 256,
        **kwargs,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Call Sentence Transformer with the given queries and passages and return the embeddings.

        Args:
            queries (Optional[List[str]]): The queries to embed.
            passages (Optional[List[str]]): The passages to embed.
            batch_size (Optional[int]): The batch size to use.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The embeddings for the queries and passages.
                If queries is None, the first element will be None.
                If passages is None, the second element will be None.
            float: The cost of the API call.
        """

        cost = 0.0
        if queries is None and passages is None:
            raise ValueError("Either queries or passages must be provided.")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if queries is not None:
                futures["queries"] = executor.submit(
                    self.process_embeddings,
                    queries,
                    batch_size,
                    "Embedding queries with OpenAI API",
                )
            if passages is not None:
                futures["passages"] = executor.submit(
                    self.process_embeddings,
                    passages,
                    batch_size,
                    "Embedding passages with OpenAI API",
                )

            queries_embeddings, passages_embeddings = None, None
            for key, future in futures.items():
                embeddings, partial_cost = future.result()
                cost += partial_cost
                if key == "queries":
                    queries_embeddings = embeddings
                else:
                    passages_embeddings = embeddings

        self.add_cost(cost)

        return (queries_embeddings, passages_embeddings), cost


class EmbeddingsSmall(Embeddings):
    def __init__(self):
        super().__init__(
            "text-embedding-3-small", pretty_name="openai_embeddings_small"
        )


class EmbeddingsLarge(Embeddings):
    def __init__(self):
        super().__init__(
            "text-embedding-3-large", pretty_name="openai_embeddings_large"
        )

        
