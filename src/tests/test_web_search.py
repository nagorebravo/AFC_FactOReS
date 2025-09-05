import logging
import unittest


class TestWebSearch(unittest.TestCase):
    logging.basicConfig(level=logging.INFO)

    @unittest.skip("Skip You.com test for now.")
    def test_you(self):
        from src.api_calls.you import You

        you = You()

        results, cost = you(
            queries=["Roman Empire"],
            top_k=2,
            language="es",
            location="us",
            download_text=True,
        )
        self.assertTrue(results)
        self.assertTrue(cost)
        keys = ["url", "favicon", "source", "text", "title", "base_url"]
        for result in results:
            for key in keys:
                self.assertTrue(key in result)

        print("YOU.COM (Roman Empire):")
        for result in results:
            print(
                f"URL: {result['url']}, Title: {result['title']}, Text: {result['text'][:50]}"
            )
        print(f"Tested YOU.COM API. Cost: {cost}$")

    def test_serpapi(self):
        from src.api_calls.serpapi import SerpAPI

        serpapi = SerpAPI()

        results, cost = serpapi(
            queries=["Roman Empire"],
            top_k=2,
            language="en",
            location="us",
            download_text=True,
        )
        self.assertTrue(results)
        self.assertTrue(cost)
        keys = ["url", "favicon", "source", "text", "title", "base_url"]
        for result in results:
            for key in keys:
                self.assertTrue(key in result)

        print("SERPAPI (Roman Empire):")
        for result in results:
            print(
                f"URL: {result['url']}, Title: {result['title']}, Text: {result['text'][:50]}"
            )
        print(f"Tested SERPAPI API. Cost: {cost}$")

    def test_newsapi(self):
        from src.api_calls.newsapi import NewsAPI

        newsapi = NewsAPI()

        results, cost = newsapi(
            queries=["Roman Empire"],
            top_k=2,
            language="en",
            location="us",
            download_text=True,
        )
        self.assertTrue(results)
        self.assertTrue(cost)
        keys = ["url", "favicon", "source", "text", "title", "base_url"]
        for result in results:
            for key in keys:
                self.assertTrue(key in result)

        print("NEWSAPI (Roman Empire):")
        for result in results:
            print(
                f"URL: {result['url']}, Title: {result['title']}, Text: {result['text'][:50]}"
            )
        print(f"Tested NEWSAPI API. Cost: {cost}$")

    def test_local(self):
        from src.api_calls.local_google import LocalGoogle

        local = LocalGoogle()

        results, cost = local(
            queries=["Roman Empire"],
            top_k=2,
            language="en",
            location="us",
            download_text=True,
        )
        self.assertTrue(results)
        self.assertEqual(cost, 0.0)
        keys = ["url", "favicon", "source", "text", "title", "base_url"]
        for result in results:
            for key in keys:
                self.assertTrue(key in result)

        print("LOCAL GOOGLE (Roman Empire):")
        for result in results:
            print(
                f"URL: {result['url']}, Title: {result['title']}, Text: {result['text'][:50]}"
            )
        print(f"Tested LOCAL GOOGLE API. Cost: {cost}$")

    def test_serper(self):
        from src.api_calls.serper import Serper

        serper = Serper()

        results, cost = serper(
            queries=["Roman Empire"],
            top_k=2,
            language="en",
            location="us",
            download_text=True,
        )
        self.assertTrue(results)
        self.assertTrue(cost)
        keys = ["url", "favicon", "source", "text", "title", "base_url"]
        for result in results:
            for key in keys:
                self.assertTrue(key in result)

        print("SERPER (Roman Empire):")
        for result in results:
            print(
                f"URL: {result['url']}, Title: {result['title']}, Text: {result['text'][:50]}"
            )
        print(f"Tested SERPER API. Cost: {cost}$")
