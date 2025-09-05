import unittest


class TestApiManager(unittest.TestCase):
    @unittest.skip("Skip test")
    def test_text_generation(self):
        import json

        from src.api_calls.api_manager import ApiManager

        api = ApiManager()

        conversation = [
            {"role": "user", "content": "What is the capital of Spain?"},
        ]
        response = api.chat(model="gpt3", conversation=conversation, max_tokens=10)
        self.assertTrue(response)
        print(f"GPT3 (Capital of Spain): {response}")
        response = api.chat(model="haiku", conversation=conversation, max_tokens=10)
        self.assertTrue(response)
        print(f"Haiku (Capital of Spain): {response}")

        total_cost = api.cost
        print(f"Tested text generation with API Manager. Cost: {total_cost}$")
        summary_cost = api.cost_summary
        print(f"Summary cost: {json.dumps(summary_cost, indent=4)}")

    def test_structured_text_generation(self):
        import json
        from pydantic import BaseModel
        from src.api_calls.api_manager import ApiManager

        api = ApiManager()

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        prompt = (
            "Extract the event information. Answer with a json with the fields name, date and participants.\n"
            "Alice and Bob are going to a science fair on Friday."
        )

        response = api.structured_completion(
            model="gpt4omini",
            prompt=prompt,
            max_tokens=128,
            pydanctic_model=CalendarEvent,
        )

        self.assertTrue(response)
        self.assertTrue(response["name"] == "Science Fair")
        self.assertTrue(response["date"] == "Friday")
        self.assertTrue(response["participants"] == ["Alice", "Bob"])
        print(f"GPT4Omini (Structured output): {response}")

        response = api.structured_completion(
            model="haiku", prompt=prompt, max_tokens=128, pydanctic_model=CalendarEvent
        )

        self.assertTrue(response)
        self.assertTrue(response["name"] == "Science Fair")
        self.assertTrue(response["date"] == "Friday")
        self.assertTrue(response["participants"] == ["Alice", "Bob"])
        print(f"Haiku (Structured output): {response}")
        summary_cost = api.cost_summary
        print(
            f"Tested structured text generation. Cost: {json.dumps(summary_cost, indent=4)}"
        )

    @unittest.skip("Skip test")
    def test_embeddings(self):
        import json

        import numpy as np

        from src.api_calls.api_manager import ApiManager

        documents = ["Un cuchillo sirve para cortar", "Un tenedor sirve para pinchar"]
        questions = ["¿Para qué sirve un cuchillo?", "¿Para qué sirve un tenedor?"]

        api = ApiManager()

        questions_embeddings_small, documents_embeddings_small = api.embeddings(
            model="openai_embeddings_small", queries=questions, passages=documents
        )
        questions_embeddings_large, documents_embeddings_large = api.embeddings(
            model="cohere_embeddings_light", queries=questions, passages=documents
        )

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

        total_cost = api.cost
        print(f"Tested embeddings with API Manager. Cost: {total_cost}$")
        summary_cost = api.cost_summary
        print(f"Summary cost: {json.dumps(summary_cost, indent=4)}")

    @unittest.skip("Skip test")
    def test_web_search(self):
        import json

        from src.api_calls.api_manager import ApiManager

        api = ApiManager()

        query = "Fact Checking"
        response = api.web_search(
            model="serpapi", queries=[query], top_k=2, language="es"
        )
        print(f"Web search (You): {[x['title'] for x in response]}")
        self.assertTrue(response)
        keys = ["url", "favicon", "source", "text", "title", "base_url"]
        for result in response:
            for key in keys:
                self.assertTrue(key in result)

        total_cost = api.cost
        print(f"Tested web search with API Manager. Cost: {total_cost}$")
        summary_cost = api.cost_summary
        print(f"Summary cost: {json.dumps(summary_cost, indent=4)}")
