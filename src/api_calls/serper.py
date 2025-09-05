import http.client
import json
import logging
import ssl
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Literal, Tuple, Optional

from src.api_calls.api_handler import ApiHandler, get_api_key
from src.utils.web_search import (
    clean_url,
    download_text_favicon_parallel,
    get_domain_name,
)

_PRICING = {
    "serper": 50 / 50000,
}


class Serper(ApiHandler):
    def __init__(self):
        super().__init__("serper")
        self.headers = {
            "X-API-KEY": get_api_key("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }

    @lru_cache(maxsize=100)
    def _get_domain_name(self, url: str) -> str:
        return get_domain_name(url)

    def _process_url(
        self, url: str, ban_domains: List[str] = None
    ) -> Optional[Dict[str, str]]: #Dict[str, str] | None:
        if url.endswith(".pdf"):
            return None
        if ban_domains and any(domain in url for domain in ban_domains):
            logging.warning(f"Excluding URL: {url} because it contains a banned domain")
            return None
        return {
            "url": url,
            "favicon": None,
            "source": self._get_domain_name(url),
            "base_url": clean_url(url),
        }

    def batch_api_call(
        self, queries: List[str], language: str, top_k: int, location: str = None
    ):
        """
        Call the serper.dev API to get search results for multiple queries

        Args:
            queries (List[str]): The queries to search for.
            language (str): The language of the search results.
            top_k (int): The number of search results to return.
            location (str): The location to search from. Defaults to None.
        Returns:
            List[List[str]]: The urls of the search results for each query.
        """

        payload = [
            {"q": query, "hl": language, "gl": location, "num": top_k}
            for query in queries
        ]
        payload = json.dumps(payload)

        # Disable SSL certificate verification
        #ssl_context = ssl._create_unverified_context()

        session = http.client.HTTPSConnection("google.serper.dev") #, context=ssl_context)
        try:
            session.request("POST", "/search", payload, self.headers)
            res = session.getresponse()
            data = json.loads(res.read().decode("utf-8"))
        finally:
            session.close()

        urls = set()
        for query_data in data:
            if "organic" in query_data:
                urls.update(result["link"] for result in query_data["organic"])

        return list(urls)

    def search(
        self,
        queries: List[str],
        language: Literal["en", "es"] = "es",
        location: str = None,
        top_k: int = 10,
        ban_domains: List[str] = None,
        download_text: bool = True,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given query using serper.dev

        Args:
            queries (List[str]): The queries to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The location to search from. Defaults to None.
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): A list of domains to exclude from the search results. Defaults to None.
            download_text (bool): Whether to download the text of the web pages. Defaults to True.
        Returns:
            List[Dict[str, str]]: The search results. Each result is a dictionary with the keys "url", "favicon", "source"
            float: The cost of the search in USD.
        """

        urls = self.batch_api_call(queries, language, top_k, location)

        with ThreadPoolExecutor() as executor:
            documents = list(
                filter(
                    None,
                    executor.map(lambda url: self._process_url(url, ban_domains), urls),
                )
            )

        if download_text:
            self._download_and_process_text(documents)

        cost = _PRICING["serper"] * len(queries)
        self.add_cost(cost)
        return documents, cost

    def __call__(
        self,
        queries: List[str],
        language: Literal["en", "es"] = "es",
        location: str = None,
        top_k: int = 10,
        ban_domains: List[str] = None,
        download_text: bool = True,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given queries using SerpAPI.

        Args:
            queries (List[str]): The queries to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The region of the search results. Defaults to None.
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): A list of domains to exclude from the search results. Defaults to None.
            download_text (bool): Whether to download the text of the web pages. Defaults to True.
        Returns:
            List[Dict[str, str]]: The search results. Each result is a dictionary with the keys "url", "favicon", "source"
            float: The cost of the search in USD.
        """

        documents, cost = self.search(
            queries=queries,
            language=language,
            location=location,
            top_k=top_k,
            ban_domains=ban_domains,
            download_text=download_text,
        )

        return documents, cost

    def _download_and_process_text(self, documents: List[Dict[str, str]]):
        urls_to_download = [x["url"] for x in documents]
        downloaded_data = download_text_favicon_parallel(urls=urls_to_download)

        for data, document in zip(downloaded_data, documents):
            if all(data):
                (
                    document["title"],
                    document["text"],
                    document["url"],
                    document["favicon"],
                ) = data

        documents[:] = [doc for doc in documents if "text" in doc]
