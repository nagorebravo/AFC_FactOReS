import unittest


class TestClaude(unittest.TestCase):
    def test_text_generation(self):
        from src.api_calls.claude import Haiku, Opus, Sonnet

        total_cost = 0.0
        haiku = Haiku()
        sonnet = Sonnet()
        opus = Opus()

        conversation = [
            {"role": "user", "content": "What is the capital of Spain?"},
        ]
        response, cost = haiku(conversation, max_tokens=35)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"Haiku (Capital of Spain): {response}")
        response, cost = sonnet(conversation, max_tokens=35)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"Sonnet (Capital of Spain): {response}")
        response, cost = opus(conversation, max_tokens=35)
        self.assertTrue(response)
        self.assertTrue(cost)
        print(f"Opus (Capital of Spain): {response}")
        print(
            f"Tested text generation with CLaude API. Cost: {haiku.cost+sonnet.cost+opus.cost}$"
        )

    def test_image_description(self):
        from src.api_calls.claude import (
            HaikuImageDescription,
            OpusImageDescription,
            SonnetImageDescription,
        )

        total_cost = 0.0

        claude_image_description = HaikuImageDescription()
        response, cost = claude_image_description(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/640px-The_Earth_seen_from_Apollo_17.jpg",
            max_tokens=35,
        )

        print(f"Haiku Vision (Earth from Apollo 17): {response}")

        total_cost += cost

        self.assertTrue(response)
        self.assertTrue(cost)

        claude_image_description = SonnetImageDescription()
        response, cost = claude_image_description(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/640px-The_Earth_seen_from_Apollo_17.jpg",
            max_tokens=35,
        )

        print(f"Sonnet Vision (Earth from Apollo 17): {response}")

        total_cost += cost

        self.assertTrue(response)
        self.assertTrue(cost)

        claude_image_description = OpusImageDescription()

        response, cost = claude_image_description(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/640px-The_Earth_seen_from_Apollo_17.jpg",
            max_tokens=35,
        )

        print(f"Opus Vision (Earth from Apollo 17): {response}")

        total_cost += cost

        self.assertTrue(response)
        self.assertTrue(cost)

        print(f"Tested image description with Claude API. Cost: {total_cost}$")
