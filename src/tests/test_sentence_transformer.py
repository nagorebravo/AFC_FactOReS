import unittest

import psutil


class TestSentenceTransformer(unittest.TestCase):
    # Skip if less than 32 GB of RAM
    @unittest.skipIf(
        psutil.virtual_memory().total < 32 * 1024 * 1024 * 1024,
        "Skipping test due to insufficient RAM",
    )
    def test_nvembeded(self):
        import torch

        from src.api_calls.sentence_transformers import NVEmbed

        total_cost = 0.0
        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]
        model = NVEmbed(quantization=True)

        (query_embeddings, passage_embeddings), cost = model(
            queries=questions,
            passages=documents,
            query_prefix="Instruct: Given a question, retrieve passages that answer the question\nQuery: ",
            passage_prefix=None,
            max_seq_length=512,
            batch_size=2,
        )

        similarity = model.similary(query_embeddings, passage_embeddings)

        answer = torch.argmax(similarity, dim=1).tolist()

        self.assertEqual(answer, [0, 1])

        print(
            f"Tested nvidia/NV-Embed-v1 embeddings with SentenceTrasnsformers. Cost: {total_cost}$"
        )

    def test_e5multilingual(self):
        import torch

        from src.api_calls.sentence_transformers import E5MultilingualLarge

        total_cost = 0.0
        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]
        model = E5MultilingualLarge(quantization=True)

        (query_embeddings, passage_embeddings), cost = model(
            queries=questions,
            passages=documents,
            query_prefix=None,
            passage_prefix=None,
            max_seq_length=512,
            batch_size=2,
        )

        similarity = model.similary(query_embeddings, passage_embeddings)

        answer = torch.argmax(similarity, dim=1).tolist()

        self.assertEqual(answer, [0, 1])

        print(
            f"Tested intfloat/multilingual-e5-large embeddings with SentenceTrasnsformers. Cost: {total_cost}$"
        )

    @unittest.skipIf(
        psutil.virtual_memory().total < 32 * 1024 * 1024 * 1024,
        "Skipping test due to insufficient RAM",
    )
    def test_e5mistral(self):
        import torch

        from src.api_calls.sentence_transformers import E5Mistral

        total_cost = 0.0
        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]
        model = E5Mistral(quantization=True)

        (query_embeddings, passage_embeddings), cost = model(
            queries=questions,
            passages=documents,
            query_prefix=None,
            passage_prefix=None,
            max_seq_length=512,
            batch_size=2,
        )

        similarity = model.similary(query_embeddings, passage_embeddings)

        answer = torch.argmax(similarity, dim=1).tolist()

        self.assertEqual(answer, [0, 1])

        print(
            f"Tested intfloat/mistral-e5-large embeddings with SentenceTrasnsformers. Cost: {total_cost}$"
        )
