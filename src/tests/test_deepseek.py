import unittest


class TestDeepseek(unittest.TestCase):
    def test_text_generation(self):
        from src.api_calls.deepseek import DeepseekChat

        deepseek = DeepseekChat()
        conversation = [
            {"role": "user", "content": "What is the capital of Spain?"},
        ]
        response, cost = deepseek(conversation, max_tokens=35)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"Deepseek-chat (Capital of Spain): {response}")

        print(f"Tested text generation with Deepseek API. Cost: {deepseek.cost}$")

    def test_structured_text_generation(self):
        from src.api_calls.deepseek import DeepseekChat
        from pydantic import BaseModel
        import json

        deepseek = DeepseekChat()

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        conversation = [
            {
                "role": "system",
                "content": "Extract the event information in json format.",
            },
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ]
        response, cost = deepseek(
            conversation, max_tokens=128, pydanctic_model=CalendarEvent
        )

        response = json.loads(response)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"DeepSeek (Structured output): {response}")
        print(
            f"Tested structured text generation with DeepSeek API. Cost: {deepseek.cost}$"
        )
