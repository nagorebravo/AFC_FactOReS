import unittest


class TestOpenAI(unittest.TestCase):
    def test_text_generation(self):
        from src.api_calls.groq import Llama38B, Llama370B, Llama405B

        llama370b = Llama370B()
        llama38b = Llama38B()
        llama405b = Llama405B()
        conversation = [
            {"role": "user", "content": "What is the capital of Spain?"},
        ]
        response, cost = llama370b(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"LLama370B (Capital of Spain): {response}")
        response, cost = llama38b(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"LLama38B (Capital of Spain): {response}")
        response, cost = llama405b(conversation, max_tokens=10)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"Llama405B (Capital of Spain): {response}")

        print(
            f"Tested text generation with OpenAI API. Cost: {llama370b.cost+llama38b.cost+llama405b.cost}$"
        )
