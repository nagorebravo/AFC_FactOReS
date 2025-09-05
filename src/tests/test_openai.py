import unittest


class TestOpenAI(unittest.TestCase):
    def test_text_generation(self):
        from src.api_calls.openai import GPT3, GPT4, GPT4O, GPT4omini

        gpt3 = GPT3()
        gpt4 = GPT4()
        gpt4o = GPT4O()
        gp4omini = GPT4omini()

        conversation = [
            {"role": "user", "content": "What is the capital of Spain?"},
        ]
        response, cost = gpt3(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"GPT3 (Capital of Spain): {response}")
        response, cost = gpt4(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"GPT4 (Capital of Spain): {response}")
        response, cost = gpt4o(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"GPT4O (Capital of Spain): {response}")
        response, cost = gp4omini(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"GPT4Omini (Capital of Spain): {response}")
        print(
            f"Tested text generation with OpenAI API. Cost: {gpt3.cost+gpt4.cost+gpt4o.cost+gp4omini.cost}$"
        )

    def test_structured_text_generation(self):
        from src.api_calls.openai import GPT4omini
        from pydantic import BaseModel

        gp4omini = GPT4omini()

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        conversation = [
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ]
        response, cost = gp4omini(
            conversation, max_tokens=128, pydanctic_model=CalendarEvent
        )
        self.assertTrue(response)
        self.assertTrue(response["name"] == "Science Fair")
        self.assertTrue(response["date"] == "Friday")
        self.assertTrue(response["participants"] == ["Alice", "Bob"])
        self.assertTrue(cost)
        print(f"GPT4Omini (Structured output): {response}")
        print(
            f"Tested structured text generation with OpenAI API. Cost: {gp4omini.cost}$"
        )

    def test_image_description(self):
        from src.api_calls.openai import GPT4ImageDescription, GPT4Vision

        total_cost = 0.0
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image below"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/640px-The_Earth_seen_from_Apollo_17.jpg",
                        },
                    },
                ],
            }
        ]

        gpt4_vision = GPT4Vision()

        response, cost = gpt4_vision(conversation, max_tokens=35)

        total_cost += cost

        self.assertTrue(response)
        print(response)
        self.assertTrue(cost)

        gpt4_image_description = GPT4ImageDescription()
        response, cost = gpt4_image_description(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/640px-The_Earth_seen_from_Apollo_17.jpg",
            max_tokens=35,
        )

        print(f"GPT4 Vision (Earth from Apollo 17): {response}")

        total_cost += cost

        self.assertTrue(response)
        self.assertTrue(cost)

        print(f"Tested image description with OpenAI API. Cost: {total_cost}$")

    def test_image_generation(self):
        from src.api_calls.openai import Dalle3

        total_cost = 0.0
        dalle3 = Dalle3()

        response, cost = dalle3(
            "A painting of a cat",
            size="1024x1024",
            quality="standard",
        )

        total_cost += cost

        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"DALL-E 3 (Cat painting): {response}")
        print(f"Tested image generation with OpenAI API. Cost: {total_cost}$")

    def test_embeddings(self):
        import numpy as np

        from src.api_calls.openai import EmbeddingsLarge, EmbeddingsSmall

        total_cost = 0.0
        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]
        embeddings_small = EmbeddingsSmall()
        embeddings_large = EmbeddingsLarge()

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

        print(f"Tested embeddings with OpenAI API. Cost: {total_cost}$")
