import datetime
import gc
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel

from src.utils.formating import extract_json

_model_to_class = {
    "haiku": "src.api_calls.claude.Haiku",
    "sonnet": "src.api_calls.claude.Sonnet",
    "opus": "src.api_calls.claude.Opus",
    "haiku_image_description": "src.api_calls.claude.HaikuImageDescription",
    "sonnet_image_description": "src.api_calls.claude.SonnetImageDescription",
    "opus_image_description": "src.api_calls.claude.OpusImageDescription",
    "gpt3": "src.api_calls.openai.GPT3",
    "gpt4": "src.api_calls.openai.GPT4",
    "gpt4o": "src.api_calls.openai.GPT4O",
    "gpt4omini": "src.api_calls.openai.GPT4omini",
    "gpt4_image_description": "src.api_calls.openai.GPT4ImageDescription",
    "gemini-flash": "src.api_calls.gemini.GeminiFlash",
    "gemini-flash-thinking": "src.api_calls.gemini.GeminiFlashThinking",
    "gemini-pro": "src.api_calls.gemini.GeminiPro",
    "dalle3": "src.api_calls.openai.Dalle3",
    "flux": "src.api_calls.replicate.Flux",
    "openai_embeddings_small": "src.api_calls.openai.EmbeddingsSmall",
    "openai_embeddings_large": "src.api_calls.openai.EmbeddingsLarge",
    "command_r": "src.api_calls.cohere.CommandR",
    "command_r_plus": "src.api_calls.cohere.CommandRPlus",
    "cohere_embeddings": "src.api_calls.cohere.EmbeddingsV3",
    "cohere_embeddings_light": "src.api_calls.cohere.EmbeddingsV3Light",
    "cohere_rerank_english": "src.api_calls.cohere.RerankEnglishV3",
    "cohere_rerank_multilingual": "src.api_calls.cohere.RerankMultilingualV3",
    "llama3-8b": "src.api_calls.groq.Llama38B",
    "llama3-70b": "src.api_calls.groq.Llama370B",
    "llama3-405b": "src.api_calls.groq.Llama405B",
    "deepseek": "src.api_calls.deepseek.DeepseekChat",
    "NV-Embed": "src.api_calls.sentence_transformers.NVEmbed",
    "E5-Multilingual-Large": "src.api_calls.sentence_transformers.E5MultilingualLarge",
    "E5-Mistral": "src.api_calls.sentence_transformers.E5Mistral",
    "you": "src.api_calls.you.You",
    "serpapi": "src.api_calls.serpapi.SerpAPI",
    "local_google": "src.api_calls.local_google.LocalGoogle",
    "newsapi": "src.api_calls.newsapi.NewsAPI",
    "serper": "src.api_calls.serper.Serper",
}

_model_to_api = {
    "haiku": "claude",
    "sonnet": "claude",
    "opus": "claude",
    "haiku_image_description": "claude",
    "sonnet_image_description": "claude",
    "opus_image_description": "claude",
    "gpt3": "openai",
    "gpt4": "openai",
    "gpt4o": "openai",
    "gpt4omini": "openai",
    "gpt4_image_description": "openai",
    "gemini-flash": "google",
    "gemini-pro": "google",
    "dalle3": "openai",
    "flux": "replicate",
    "openai_embeddings_small": "openai",
    "openai_embeddings_large": "openai",
    "command_r": "cohere",
    "command_r_plus": "cohere",
    "cohere_embeddings": "cohere",
    "cohere_embeddings_light": "cohere",
    "cohere_rerank_english": "cohere",
    "cohere_rerank_multilingual": "cohere",
    "deepseek": "deepseek",
    "llama3-8b": "groq",
    "llama3-70b": "groq",
    "llama3-405b": "groq",
    "you": "you",
    "serpapi": "serpapi",
    "local_google": "local",
    "newsapi": "newsapi",
    "serper": "serper",
    "NV-Embed": "sentence_transformers",
    "E5-Multilingual-Large": "sentence_transformers",
    "E5-Mistral": "sentence_transformers",
}


class ApiManager:
    def __init__(self):
        self.api_handlers = {}
        self._call_history = []  # Cost of each API call

        

    @property
    def total_cost(self) -> float:
        """
        Returns the total cost of all API calls

        Returns:
            float: The total cost of all API calls
        """
        return sum([call["cost"] for call in self._call_history])

    @property
    def cost_per_model(self) -> List[Dict[str, float]]:
        """
        Returns the cost of each API call

        Returns:
            List[Dict[str, float]]: The cost of each API call
        """
        cost_per_api_dict = {}
        for model in self.api_handlers:
            api_handler = self.api_handlers[model]
            model_name = api_handler.api_name
            if model_name not in cost_per_api_dict:
                cost_per_api_dict[model_name] = 0.0
            cost_per_api_dict[model_name] += api_handler.cost
        return cost_per_api_dict

    @property
    def cost_per_api(self) -> Dict[str, float]:
        """
        Returns the cost of each API

        Returns:
            Dict[str, float]: The cost of each API
        """
        cost_per_api_dict = {}
        for model in self.api_handlers:
            api_handler = self.api_handlers[model]
            model_name = api_handler.api_name
            api_name = _model_to_api[model_name]
            if api_name not in cost_per_api_dict:
                cost_per_api_dict[api_name] = 0.0
            cost_per_api_dict[api_name] += api_handler.cost
        return cost_per_api_dict

    @property
    def num_calls(self) -> int:
        """
        Returns the number of API calls

        Returns:
            int: The number of API calls
        """
        return len(self._call_history)

    @property
    def call_history(self) -> List[Dict[str, Any]]:
        """
        Returns the history of the API calls

        Returns:
            List[Dict[str, Any]]: The history of the API calls
        """
        return self._call_history

    @property
    def cost(self) -> float:
        """
        Returns the total cost of all API calls

        Returns:
            float: The total cost of all API calls
        """
        return self.total_cost

    @property
    def cost_summary(self) -> Dict[str, float]:
        """
        Returns a summary of the cost of the API calls

        Returns:
            Dict[str, float]: A summary of the cost of the API calls
        """
        return {
            "total_cost": self.total_cost,
            "cost_per_api": self.cost_per_api,
            "cost_per_model": self.cost_per_model,
            "num_calls": self.num_calls,
            "call_history": self.call_history,
        }

    @property
    def last_call_cost(self) -> float:
        """
        Returns the cost of the last API call

        Returns:
            float: The cost of the last API call
        """
        if len(self._call_history) == 0:
            return 0.0
        return self._call_history[-1]["cost"]

    def reset_cost(self):
        self._call_history = []
        for model in self.api_handlers:
            self.api_handlers[model].reset_cost()

    def chat(
        self,
        model,
        conversation,
        max_tokens: int = None,
        temperature: float = None,
        **kwargs,
    ):
        """
        Call the chat API with the given conversation and return the response.

        Args:
            model (str): The name of the model to call.
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[int]): The temperature to use.

        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """
        return self.__call__(
            model,
            conversation=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def completion(
        self, model, prompt, max_tokens: int = None, temperature: float = None, **kwargs
    ):
        """
        Call the chat API with the given conversation and return the response.

        Args:
            model (str): The name of the model to call.
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[int]): The temperature to use.

        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """

        conversation = [{"role": "user", "content": prompt}]

        return self.__call__(
            model,
            conversation=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    def structured_completion(
        self,
        model,
        prompt,
        pydantic_model: BaseModel,
        max_tokens: int = None,
        temperature: float = None,
        MAX_RETRIES: int = 3,
        **kwargs,
    ):
        """
        Call the chat API with the given conversation and return the response.

        Args:
            model (str): The name of the model to call.
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
            temperature (Optional[int]): The temperature to use.

        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """

        conversation = [{"role": "user", "content": prompt}]

        response = self.__call__(
            model,
            conversation=conversation,
            max_tokens=max_tokens,
            temperature=temperature,
            pydantic_model=pydantic_model,
            **kwargs,
        )

        if not isinstance(response, str):
            return response
        else:
            for _ in range(MAX_RETRIES):
                try:
                    parsed_response = extract_json(
                        response, fields=list(pydantic_model.model_fields.keys())
                    )
                    return parsed_response
                except Exception as e:
                    logging.warning(f"Failed to parse response: {e}")
                    response = self.__call__(
                        model,
                        conversation=conversation,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        pydantic_model=pydantic_model,
                        **kwargs,
                    )

            raise ValueError(f"Failed to parse response after {MAX_RETRIES} attempts")

    def image_description(self, model, image_url, max_tokens: int = None):
        """
        Call the image description API with the given image and return the response.

        Args:
            model (str): The name of the model to call.
            image_url (str): The URL of the image to send to the API.
            max_tokens (Optional[int]): The maximum number of tokens to generate.

        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """
        return self.__call__(model, image_url=image_url, max_tokens=max_tokens)

    def image_generation(
        self, model, prompt, size: str = "1024x1024", quality: str = "standard"
    ):
        """
        Call the image generation API with the given prompt and return the response.

        Args:
            model (str): The name of the model to call.
            prompt (str): The prompt to send to the API.
            size (str): The size of the image to generate.
            quality (str): The quality of the image to generate.

        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """
        return self.__call__(
            model, image_description=prompt, size=size, quality=quality
        )

    def embeddings(
        self,
        model: str,
        queries: Optional[List[str]] = None,
        passages: Optional[List[str]] = None,
        query_prefix: Optional[str] = None,
        passage_prefix: Optional[str] = None,
        max_seq_length: Optional[int] = 1024,
        batch_size: Optional[int] = 256,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Call the embeddings API with the given queries and passages and return the embeddings.

        Args:
            model (str): The name of the model to call.
            queries (Optional[List[str]]): The queries to embed.
            passages (Optional[List[str]]): The passages to embed.
            query_prefix (Optional[str]): The prompt to use for the queries.
            passage_prefix (Optional[str]): The prompt to use for the passages.
            max_seq_length (Optional[int]): The maximum sequence length to use.
            batch_size (Optional[int]): The batch size to use.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The embeddings for the queries and passages.
                If queries is None, the first element will be None.
                If passages is None, the second element will be None.
            float: The cost of the API call.
        """
        return self.__call__(
            model,
            queries=queries,
            passages=passages,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )

    def rerank(
        self,
        model: str,
        queries: Optional[List[str]] = None,
        passages: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Call the Rerank API with the given queries and passages and return the ranks.

        Args:
            model (str): The name of the model to call.
            queries (Optional[List[str]]): The queries to embed.
            passages (Optional[List[str]]): The passages to embed.
        Returns:
            torch.Tensor: The rank for each passage for each query.
            float: The cost of the API call.
        """
        return self.__call__(
            model,
            queries=queries,
            passages=passages,
        )

    def web_search(
        self,
        model: str,
        queries: List[str],
        language: str = "es",
        location: str = None,
        top_k: int = 10,
        ban_domains: List[str] = None,
        download_text: bool = True,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given queries.

        Args:
            model (str): The name of the model to call.
            queries (List[str]): The queries to search for.
            country (str): The language code of the search results. Defaults to "es".
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): The domains to exclude from the search results.
            download_text (bool): Whether to download the text of the web pages.

        Returns:
            (List[Dict[str, str]], float): The search results and the cost of the search in USD.
            The Dict has the keys "url", "favicon", "source", "text", "title", "base_url"
        """

        return self.__call__(
            model,
            queries=queries,
            language=language,
            location=location,
            top_k=top_k,
            ban_domains=ban_domains,
            download_text=download_text,
        )

    def news_search(
        self,
        queries: List[str],
        language: str = "es",
        location: str = None,
        top_k: int = 10,
        ban_domains: List[str] = None,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        download_text: bool = True,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search for news articles on the internet.

        Args:
            queries (List[str]): The queries to search for.
            country (str): The language code of the search results. Defaults to "es".
            location (str): The region of the search results. Defaults to None.
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): The domains to exclude from the search results.
            start_date (datetime.datetime): The start date of the search results.
            end_date (datetime.datetime): The end date of the search results.
            download_text (bool): Whether to download the text of the web pages.

        Returns:
            (List[Dict[str, str]], float): The search results and the cost of the search in USD.
            The Dict has the keys "url", "favicon", "source", "text", "title", "base_url"

        """

        return self.__call__(
            "serpapi",
            queries=queries,
            language=language,
            location=location,
            top_k=top_k,
            ban_domains=ban_domains,
            start_date=start_date,
            end_date=end_date,
            download_text=download_text,
            search_news=True,
        )

    def __call__(self, model, **kwargs) -> Any:
        """
        Call the API with the given model and arguments and return the response.

        Args:
            model (str): The name of the model to call.
            **kwargs: The arguments to pass to the API.

        Returns:
            Any: The response from the API.
        """
        if model not in _model_to_class:
            raise NotImplementedError(f"Model {model} not implemented")

        if model not in self.api_handlers:
            # Lazy import the model class
            module_path, class_name = _model_to_class[model].rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            model_class = getattr(module, class_name)
            self.api_handlers[model] = model_class()

        # Filtrar kwargs no compatibles ########################
        # valid_kwargs = {k: v for k, v in kwargs.items() if k not in ["proxies"]}

        api_handler = self.api_handlers[model]
        response, cost = api_handler(**kwargs)
        self._call_history.append({"id": self.num_calls, "model": model, "cost": cost})
        return response

    def close_api(self, model: str):
        """
        Close the API connection for the given model.
        This is useful for local models that need to be closed after use to free GPU memory.

        Args:
            model (str): The name of the model to close
        """
        if model in self.api_handlers:
            del self.api_handlers[model]
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Model {model} not found in api_handlers")
