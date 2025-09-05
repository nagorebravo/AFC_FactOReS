import unittest


class TestReplicate(unittest.TestCase):
    def test_flux(self):
        from src.api_calls.replicate import Flux

        total_cost = 0.0
        flux = Flux()

        response, cost = flux(
            "A painting of a cat",
            size="1024x1024",
        )

        total_cost += cost

        self.assertTrue(response)
        self.assertTrue(total_cost)
        print(f"Flux (Cat painting): {response}")
        print(f"Tested image generation with Flux API. Cost: {total_cost}$")
