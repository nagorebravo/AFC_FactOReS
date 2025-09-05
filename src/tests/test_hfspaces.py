import unittest


class TestHFSpaces(unittest.TestCase):
    

    def test_flux(self):
        from src.api_calls.hf_spaces import Flux

        total_cost = 0.0
        flux = Flux()

        response, cost = flux(
            "A painting of a cat",
            size="1024x1024",
        )

        total_cost += cost

        self.assertTrue(response)
        self.assertEqual(total_cost, 0.0)
        print(f"Flux (Cat painting): {response}")
        print(f"Tested image generation with Flux API. Cost: {total_cost}$")