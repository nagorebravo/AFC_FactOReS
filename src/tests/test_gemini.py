import unittest


class TestGoogle(unittest.TestCase):
    def test_text_generation(self):
        from src.api_calls.gemini import GeminiFlash, GeminiPro, GeminiFlashThinking

        gemini_flash = GeminiFlash()
        gemini_pro = GeminiPro()
        gemini_flash_thinking = GeminiFlashThinking()

        conversation = [
            {"role": "user", "content": "What is the capital of Spain?"},
        ]
        response, cost = gemini_flash(conversation, max_tokens=35)
        self.assertTrue(response)
        # self.assertTrue(cost)
        print(f"Gemini Flash (Capital of Spain): {response}")
        response, cost = gemini_pro(conversation, max_tokens=35)
        self.assertTrue(response)
        # self.assertTrue(cost)
        print(f"Gemini Pro (Capital of Spain): {response}")

        response, cost = gemini_flash_thinking(conversation, max_tokens=35)
        self.assertTrue(response)
        # self.assertTrue(cost)
        print(f"Gemini Flash Thinking (Capital of Spain): {response}")

        # Test system prompt
        conversation = [
            {"role": "system", "content": "You are an expert in geography."},
            {"role": "user", "content": "What is the capital of Spain?"},
        ]

        response, cost = gemini_flash(conversation, max_tokens=35)
        self.assertTrue(response)
        # self.assertTrue(cost)
        print(f"Gemini Flash (Capital of Spain w/system prompt): {response}")

        print(
            f"Tested text generation with Gemini API. Cost: {gemini_flash.cost+gemini_pro.cost}$"
        )

    def test_structured_text_generation(self):
        from src.api_calls.gemini import GeminiFlash
        from pydantic import BaseModel

        gemini_flash = GeminiFlash()

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
        response, cost = gemini_flash(
            conversation, max_tokens=128, pydanctic_model=CalendarEvent
        )

        self.assertTrue(response)
        self.assertTrue(isinstance(response, dict))
        # self.assertTrue(cost)
        print(f"Gemini (Structured output): {response}")
        print(
            f"Tested structured text generation with Gemini API. Cost: {gemini_flash.cost}$"
        )

    def test_web_search(self):
        from src.api_calls.gemini import GeminiFlash

        gemini_flash = GeminiFlash()

        conversation = [
            {"role": "user", "content": "How many people live in Spain?"},
        ]
        response, cost = gemini_flash(conversation, max_tokens=35, web_search=True)
        self.assertTrue(response)
        # self.assertTrue(cost)
        print(f"Gemini Flash (Web Search): {response}")
