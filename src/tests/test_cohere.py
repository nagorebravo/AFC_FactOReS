import unittest


class TestCohere(unittest.TestCase):
    def test_text_generation(self):
        from src.api_calls.cohere import CommandR, CommandRPlus

        commandr = CommandR()
        commandrplus = CommandRPlus()
        conversation = [
            {"role": "user", "content": "What is the capital of Spain?"},
        ]
        response, cost = commandr(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"CommandR (Capital of Spain): {response}")
        response, cost = commandrplus(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"CommandR-Plus (Capital of Spain): {response}")
        print(
            f"Tested text generation with Cohere API. Cost: {commandr.cost+commandrplus.cost}$"
        )

    def test_embeddings(self):
        import numpy as np

        from src.api_calls.cohere import EmbeddingsV3, EmbeddingsV3Light

        total_cost = 0.0
        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]
        embeddings_small = EmbeddingsV3Light()
        embeddings_large = EmbeddingsV3()

        (questions_embeddings_small, documents_embeddings_small), cost = (
            embeddings_small(queries=questions, passages=documents)
        )
        total_cost += cost
        (questions_embeddings_large, documents_embeddings_large), cost = (
            embeddings_large(queries=questions, passages=documents)
        )
        total_cost += cost

        def cosine_similarity(a: np.array, b: np.array) -> np.array:
            """
            Compute the cosine similarity between the given embeddings.

            Args:
                a (np.ndarray): The first set of embeddings.
                b (np.ndarray): The second set of embeddings.

            Returns:
                np.ndarray: The cosine similarity between the given embeddings.
            """

            norms: np.ndarray = np.sqrt(np.sum(a**2, axis=1))
            norms[norms == 0] = 1
            na: np.ndarray = a / norms[:, np.newaxis]

            norms: np.ndarray = np.sqrt(np.sum(b**2, axis=1))
            norms[norms == 0] = 1
            nb: np.ndarray = b / norms[:, np.newaxis]

            return na.dot(nb.T)

        sim_small = cosine_similarity(
            np.asarray(questions_embeddings_small),
            np.asarray(documents_embeddings_small),
        )
        sim_large = cosine_similarity(
            np.asarray(questions_embeddings_large),
            np.asarray(documents_embeddings_large),
        )

        answer_small = np.argmax(sim_small, axis=1).tolist()
        answer_large = np.argmax(sim_large, axis=1).tolist()

        self.assertTrue(answer_small == [0, 1])
        self.assertEqual(answer_large, [0, 1])

        print(f"Tested embeddings with Cohere API. Cost: {total_cost}$")

    def test_rerank(self):
        import torch

        from src.api_calls.cohere import RerankMultilingualV3

        total_cost = 0.0
        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]
        reranker = RerankMultilingualV3()

        ranks, cost = reranker(queries=questions, passages=documents)
        total_cost += cost

        answer = torch.argmax(ranks, axis=1).tolist()

        self.assertEqual(answer, [0, 1])

        print(f"Tested reranking with Cohere API. Cost: {total_cost}$")
