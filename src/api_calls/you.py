import json
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Literal, Tuple

import requests

from src.api_calls.api_handler import ApiHandler, get_api_key
from src.utils.web_search import (
    clean_url,
    download_favicons,
    download_text_favicon_parallel,
    get_domain_name,
)

_PRICING = {
    "you": 100 / 11765,
}


class You(ApiHandler):
    def __init__(self):
        super().__init__("you")

    def search(
        self,
        query,
        language: Literal["en", "es"] = "es",
        location: str = None,
        top_k=10,
        ban_domains: List[str] = None,
        get_favicons: bool = True,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given query using YOU.COM.

        Args:
            query (str): The query to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The location to search from. Defaults to None.
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): A list of domains to exclude from the search results. Defaults to None.
            get_favicons (bool): Whether to download the favicons of the search results. Defaults to True.
        Returns:
            List[Dict[str, str]]: The search results. Each result is a dictionary with the keys "url", "favicon", "source"
            float: The cost of the search in USD.
        """

        if language == "en":
            language = "US"
        elif language == "es":
            language = "ES"
        else:
            raise ValueError(
                f"Unsupported language: {language}. Supported languages are 'es' and 'en'."
            )

        querystring = {
            "query": query,
            "num_web_results": top_k,
            "country": language.upper(),
        }
        headers = {"X-API-Key": get_api_key("YOU_API_KEY")}
        response = requests.request(
            "GET",
            "https://api.ydc-index.io/search",
            headers=headers,
            params=querystring,
        )

        results = json.loads(response.text.encode("utf8"))

        documents = []
        if "hits" not in results:
            logging.warning(
                f"Skipping query: {query} because there are no search results.\n"
                f"Results: {results}"
            )
            cost = _PRICING["you"]
            self.add_cost(cost)
            return documents, cost

        favicon_to_download = []
        ban_domains_set = set(ban_domains) if ban_domains else set()

        for result in results.get("hits", []):
            if result.get("url", "").endswith(".pdf"):
                continue

            if any(domain in result.get("url", "") for domain in ban_domains_set):
                continue

            if "url" not in result or "title" not in result or "snippets" not in result:
                logging.warning(
                    f"Skipping result: {result} because it is missing required fields."
                )
                continue

            if result["snippets"] is None or len(result["snippets"]) == 0:
                logging.warning(
                    f"Skipping result: {result} because it has no snippets."
                )
                continue

            source = get_domain_name(result["url"])
            favicon = (
                "https://upload.wikimedia.org/wikipedia/commons/0/01/Website_icon.svg"
            )
            text = "\n".join(result["snippets"])
            title = result["title"]
            url = result["url"]
            base_url = clean_url(url)
            documents.append(
                {
                    "url": url,
                    "favicon": favicon,
                    "source": source,
                    "text": text,
                    "title": title,
                    "base_url": base_url,
                }
            )

            favicon_to_download.append(base_url)

        if get_favicons:
            favicons = download_favicons(favicon_to_download)
            for document in documents:
                base_url = document["base_url"]
                if base_url in favicons:
                    document["favicon"] = favicons[base_url]

        cost = _PRICING["you"]
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
        Search the internet for the given queries using YOU.COM.

        Args:
            queries (List[str]): The queries to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The location to search from. Defaults to None.
            top_k (int): The number of search results to return for each query. Defaults to 10.
            ban_domains (List[str]): A list of domains to exclude from the search results. Defaults to None.
            download_text (bool): Whether to download the text of the search results. Defaults to True.

        Returns:
            List[Dict[str, str]]: The search results. Each result is a dictionary with the keys "url", "favicon", "source"
            float: The cost of the search in USD.
        """

        with ThreadPoolExecutor() as executor:
            search_func = partial(
                self.search,
                language=language,
                location=location,
                top_k=top_k,
                ban_domains=ban_domains,
                get_favicons=False,
            )
            results = list(executor.map(search_func, queries))

        all_documents = []
        total_cost = 0.0

        for documents, cost in results:
            all_documents.extend(documents)
            total_cost += cost

        # Deduplicate the URLs
        all_documents = list({doc["url"]: doc for doc in all_documents}.values())

        if not download_text:
            favicon_to_download = [document["base_url"] for document in all_documents]
            favicons = download_favicons(favicon_to_download)
            for document in all_documents:
                base_url = document["base_url"]
                if base_url in favicons:
                    document["favicon"] = favicons[base_url]
        else:
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
