import concurrent.futures
import os
from typing import Dict, List, Optional, Tuple

import cohere
import torch
from tqdm.auto import tqdm

from src.api_calls.api_handler import ApiHandler, get_api_key
from src.api_calls.utils import batch

_PRICING = {
    "reranker": 2 / 1000,
    "embedding": 0.10 / 1000000,
    "command-r_input": 0.15 / 1000000,
    "command-r_output": 0.60 / 1000000,
    "command-r-plus_input": 2.5 / 1000000,
    "command-r-plus_output": 10.0 / 1000000,
}

_roles_dict = {
    "user": "USER",
    "assistant": "ASSISTANT",
    "system": "SYSTEM",
}


class CommandRBase(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = cohere.Client(api_key=get_api_key("COHERE_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def replace_role(self, role: str) -> str:
        """
        Replace the role of the message to match the API format.

        Args:
            role (str): The role to replace.

        Returns:
            str: The role replaced. If the role is not in the dictionary, it returns the same role.
        """
        if role in _roles_dict:
            return _roles_dict[role]
        return role

    def format_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """
        Replace the roles of the conversation to match the API format.

        Args:
            conversation (List[Dict[str, str]]): The conversation to replace the roles.

        Returns:
            List[Dict[str, str]]: The conversation with the roles replaced.
        """
        return [
            {
                "role": self.replace_role(message["role"]),
                "message": message["content"],
            }
            for message in conversation
        ]

    def __call__(
        self,
        conversation: List[Dict[str, str]],
        temperature: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[str, float]:
        """
        Call the COHERE API with the given conversation and return the response.

        Args:
            conversation (List[Dict[str, str]]): The conversation to send to the API.
            temperature (Optional[int]): The temperature to use.
            max_tokens (Optional[int]): The maximum number of tokens to generate.
        Returns:
            Tuple[str, float]: The response from the API and the cost of the call.
        """
        conversation = self.format_conversation(conversation)
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        if conversation[0]["role"] == "SYSTEM":
            system_prompt = conversation[0]["message"]
            conversation = conversation[1:]
            kwargs["preamble"] = system_prompt

        message = conversation[-1]["message"]
        conversation = conversation[:-1]

        response = self.client.chat(
            model=self.model,
            message=message,
            chat_history=conversation,
            **kwargs,
        )

        cost = (
            response.meta.billed_units.input_tokens * _PRICING[f"{self.model}_input"]
            + response.meta.billed_units.output_tokens
            * _PRICING[f"{self.model}_output"]
        )

        self.add_cost(cost)

        return response.text, cost


class CommandR(CommandRBase):
    def __init__(self):
        super().__init__("command-r", pretty_name="command_r")


class CommandRPlus(CommandRBase):
    def __init__(self):
        super().__init__("command-r-plus", pretty_name="command_r_plus")


class Embeddings(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        queries: Optional[List[str]] = None,
        passages: Optional[List[str]] = None,
        batch_size: Optional[int] = 128,
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

        if queries is not None:
            queries_embeddings = []
            batches = list(batch(queries, batch_size))
            for q in tqdm(
                batches,
                desc="Embedding queries with COHERE API",
                disable=len(batches) == 1,
            ):
                try:
                    response = self.client.embed(
                        texts=q,
                        model=self.model,
                        input_type="search_query",
                    )
                    cost += (
                        response.meta.billed_units.input_tokens * _PRICING["embedding"]
                    )
                    queries_embeddings.extend(response.embeddings)
                except Exception:
                    for q_ in q:
                        try:
                            response = self.client.embed(
                                texts=[q_],
                                model=self.model,
                                input_type="search_query",
                            )
                            cost += response.usage.total_tokens * _PRICING[self.model]
                            queries_embeddings.extend(response.embeddings)
                        except Exception as e:
                            raise ValueError(f"Error processing query: {q_}\n{e}")
            queries_embeddings = torch.tensor(queries_embeddings)

        else:
            queries_embeddings = None

        if passages is not None:
            passages_embeddings = []
            batches = list(batch(passages, batch_size))
            for p in tqdm(
                batches,
                desc="Embedding passages with COHERE API",
                disable=len(batches) == 1,
            ):
                try:
                    response = self.client.embed(
                        texts=p,
                        model=self.model,
                        input_type="search_document",
                    )
                    cost += (
                        response.meta.billed_units.input_tokens * _PRICING["embedding"]
                    )
                    passages_embeddings.extend(response.embeddings)
                except Exception:
                    for p_ in p:
                        try:
                            response = self.client.embed(
                                texts=[p_],
                                model=self.model,
                                input_type="search_document",
                            )
                            cost += response.usage
                            passages_embeddings.extend(response.embeddings)
                        except Exception as e:
                            raise ValueError(f"Error processing passage: {p_}\n{e}")
            passages_embeddings = torch.tensor(passages_embeddings)

        else:
            passages_embeddings = None

        self.add_cost(cost)

        return (queries_embeddings, passages_embeddings), cost


class EmbeddingsV3(Embeddings):
    def __init__(self):
        super().__init__("embed-multilingual-v3.0", pretty_name="cohere_embeddings")


class EmbeddingsV3Light(Embeddings):
    def __init__(self):
        super().__init__(
            "embed-multilingual-light-v3.0", pretty_name="cohere_embeddings_light"
        )


class Reranker(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None):
        self.model = model
        self.client = cohere.Client(api_key=get_api_key("COHERE_API_KEY"))
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        queries: List[str],
        passages: List[str],
        max_workers: int = os.cpu_count() * 8,
        **kwargs,
    ) -> Tuple[torch.Tensor, float]:
        """
        Call Cohere Reranker with the given queries and passages in parallel and return the rankings.

        Args:
            queries (List[str]): The queries to rerank against.
            passages (List[str]): The passages to rerank.
            max_workers (int): Maximum number of parallel workers.
        Returns:
            torch.Tensor: The rank for each passage for each query.
            float: The cost of the API call.
        """
        if not queries or not passages:
            raise ValueError(
                "Both queries and passages must be provided and non-empty."
            )

        ranks = torch.zeros((len(queries), len(passages)), dtype=torch.float32)
        total_cost = 0.0

        def process_query(args):
            i, query = args
            scores = self.client.rerank(
                model=self.model,
                query=query,
                documents=passages,
                top_n=len(passages),
            )
            cost = scores.meta.billed_units.search_units * _PRICING["reranker"]
            query_ranks = torch.zeros(len(passages), dtype=torch.float32)
            for score in scores.results:
                query_ranks[score.index] = score.relevance_score
            return i, query_ranks, cost

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(process_query, (i, query)): i
                for i, query in enumerate(queries)
            }
            for future in concurrent.futures.as_completed(future_to_query):
                i, query_ranks, cost = future.result()
                ranks[i] = query_ranks
                total_cost += cost

        self.add_cost(total_cost)
        return ranks, total_cost


class RerankEnglishV3(Reranker):
    def __init__(self):
        super().__init__("rerank-english-v3.0", pretty_name="cohere_rerank_english")


class RerankMultilingualV3(Reranker):
    def __init__(self):
        super().__init__(
            "rerank-multilingual-v3.0", pretty_name="cohere_rerank_multilingual"
        )
