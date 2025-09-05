import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Literal, Tuple

from googlesearch import search

from src.api_calls.api_handler import ApiHandler
from src.utils.web_search import (
    clean_url,
    download_text_favicon_parallel,
    get_domain_name,
)

_PRICING = {
    "google": 0,
}


class LocalGoogle(ApiHandler):
    def __init__(self):
        super().__init__("local_google")

    @lru_cache(maxsize=100)
    def _get_domain_name(self, url: str) -> str:
        return get_domain_name(url)

    def _process_url(
        self, url: str, ban_domains: List[str] = None
    ) -> Dict[str, str] | None:
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

    def search(
        self,
        query: str,
        language: Literal["en", "es"] = "es",
        location: str = None,
        top_k: int = 10,
        ban_domains: List[str] = None,
        download_text: bool = True,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given query using Local Google Serach.

        Args:
            query (str): The query to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The location to search from. Defaults to None.
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): A list of domains to exclude from the search results. Defaults to None.
            download_text (bool): Whether to download the text of the web pages. Defaults to True.
        Returns:
            List[Dict[str, str]]: The search results. Each result is a dictionary with the keys "url", "favicon", "source"
            float: The cost of the search in USD.
        """

        params = (
            {"lang": language, "region": location} if location else {"lang": language}
        )
        urls: List[str] = list(search(term=query, **params, num_results=top_k))

        if not urls:
            logging.warning(
                f"No search results found for the query: {query}. Results: {urls}"
            )
            return [], 0.0

        with ThreadPoolExecutor() as executor:
            documents = list(
                filter(
                    None,
                    executor.map(lambda url: self._process_url(url, ban_domains), urls),
                )
            )

        if download_text:
            self._download_and_process_text(documents)
        self.add_cost(0.0)
        return documents, 0.0

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

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.search,
                    query=query,
                    language=language,
                    location=location,
                    top_k=top_k,
                    ban_domains=ban_domains,
                    download_text=False,
                )
                for query in queries
            ]

            all_documents = []
            total_cost = 0.0

            for future in as_completed(futures):
                documents, cost = future.result()
                all_documents.extend(documents)
                total_cost += cost

        # Deduplicate the URLs
        all_documents = list({doc["url"]: doc for doc in all_documents}.values())

        if download_text:
            self._download_and_process_text(all_documents)

        return all_documents, total_cost

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
