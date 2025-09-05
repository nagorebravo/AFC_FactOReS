import torch
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
from transformers import BitsAndBytesConfig
from src.api_calls.api_handler import ApiHandler


class SentenceTransformerBase(ApiHandler):
    def __init__(self, model: str, pretty_name: str = None, quantization: bool = False):
        self.model = model

        if quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_args = {
                "quantization_config": quantization_config,
            }
        else:
            model_args = {}

        self.client = SentenceTransformer(
            model, trust_remote_code=True, model_kwargs=model_args
        )
        super().__init__(model if pretty_name is None else pretty_name)

    def __call__(
        self,
        queries: Optional[List[str]] = None,
        passages: Optional[List[str]] = None,
        query_prefix: Optional[str] = None,
        passage_prefix: Optional[str] = None,
        max_seq_length: Optional[int] = 1024,
        batch_size: Optional[int] = 2,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Call Sentence Transformer with the given queries and passages and return the embeddings.

        Args:
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

        self.client.max_seq_length = max_seq_length

        if queries is None and passages is None:
            raise ValueError("Either queries or passages must be provided.")

        if queries is not None:
            query_embeddings = self.client.encode(
                sentences=queries,
                batch_size=batch_size,
                prompt=query_prefix,
                normalize_embeddings=True,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=True,
            ).cpu()
        else:
            query_embeddings = None

        if passages is not None:
            passage_embeddings = self.client.encode(
                sentences=passages,
                batch_size=batch_size,
                prompt=passage_prefix,
                normalize_embeddings=True,
                convert_to_numpy=False,
                convert_to_tensor=True,
                show_progress_bar=True,
            ).cpu()
        else:
            passage_embeddings = None

        return (query_embeddings, passage_embeddings), 0.0

    def row_wise_similarity(
        self, query_embeddings: torch.Tensor, passage_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the row-wise similarity between the query and passage embeddings.

        Args:
            query_embeddings (torch.Tensor): The query embeddings.
            passage_embeddings (torch.Tensor): The passage embeddings.
        Returns:
            torch.Tensor: The row-wise similarity between the query and passage embeddings.
        """
        if torch.cuda.is_available():
            query_embeddings = query_embeddings.to("cuda")
            passage_embeddings = passage_embeddings.to("cuda")

        with torch.no_grad():
            similarity = torch.sum(query_embeddings * passage_embeddings, dim=1)
        return similarity.cpu()

    def similary(
        self, query_embeddings: torch.Tensor, passage_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the similarity between each query and passage embeddings.

        Args:
            query_embeddings (torch.Tensor): The query embeddings.
            passage_embeddings (torch.Tensor): The passage embeddings.
        Returns:
            torch.Tensor: The similarity between each query and passage embeddings.
        """
        if torch.cuda.is_available():
            query_embeddings = query_embeddings.to("cuda")
            passage_embeddings = passage_embeddings.to("cuda")

        with torch.no_grad():
            similarity = query_embeddings @ passage_embeddings.T
        return similarity.cpu()


class NVEmbed(SentenceTransformerBase):
    def __init__(self, quantization: bool = True):
        super().__init__(
            "nvidia/NV-Embed-v1", pretty_name="NV-Embed", quantization=quantization
        )

    def __call__(
        self,
        queries: Optional[List[str]] = None,
        passages: Optional[List[str]] = None,
        query_prefix: Optional[str] = None,
        passage_prefix: Optional[str] = None,
        max_seq_length: Optional[int] = 1024,
        batch_size: Optional[int] = 2,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
        """
        Call Sentence Transformer with the given queries and passages and return the embeddings.

        Args:
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

        self.client.tokenizer.padding_side = "right"

        return super().__call__(
            queries=self.add_eos(queries) if queries is not None else None,
            passages=self.add_eos(passages) if passages is not None else None,
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
        )

    def add_eos(self, input_examples):
        input_examples = [
            input_example + self.client.tokenizer.eos_token
            for input_example in input_examples
        ]
        return input_examples


class E5Mistral(SentenceTransformerBase):
    def __init__(self, quantization: bool = True):
        super().__init__(
            "intfloat/e5-mistral-7b-instruct",
            pretty_name="E5-Mistral",
            quantization=quantization,
        )


class E5MultilingualLarge(SentenceTransformerBase):
    def __init__(self, quantization: bool = False):
        super().__init__(
            "intfloat/multilingual-e5-large",
            pretty_name="E5-Multilingual-Large",
            quantization=quantization,
        )


class E5MultilingualLargeFinetuned(SentenceTransformerBase):
    def __init__(self, quantization: bool = False):
        super().__init__(
            "results/models/E5-Multilingual-Large",
            pretty_name="results/models/E5-Multilingual-Large",
            quantization=quantization,
        )
