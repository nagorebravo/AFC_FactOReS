import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, Tuple

from newsapi import NewsApiClient

from src.api_calls.api_handler import ApiHandler, get_api_key
from src.utils.web_search import clean_url, download_text_favicon_parallel

_PRICING = {
    "newsapi": 449 / 250000,
}


class NewsAPI(ApiHandler):
    def __init__(self):
        self.client = NewsApiClient(api_key=get_api_key("NEWS_API_KEY"))

        super().__init__("newsapi")

    def search(
        self,
        query: str,
        language: Literal["en", "es"] = "es",
        location: str = None,
        top_k: int = 10,
        ban_domains: List[str] = None,
        download_text: bool = True,
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given query using News API.

        Args:
            query (str): The query to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The location to search from. Defaults to None.
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): A list of domains to exclude from the search results. Defaults to None.
            download_text (bool): Whether to download the text of the web pages. Defaults to True.
            start_date (datetime.datetime): The start date of the search results. Defaults to None.
            end_date (datetime.datetime): The end date of the search results. Defaults to None.
        Returns:
            List[Dict[str, str]]: The search results. Each result is a dictionary with the keys "url", "favicon", "source"
            float: The cost of the search in USD.
        """
        print(
            f"q:{query}, language:{language}, page_size:{top_k}, from_param:{start_date}, to:{end_date}, exclude_domains:{ban_domains}"
        )
        results = self.client.get_everything(
            q=query,
            language=language,
            page_size=top_k,
            from_param=start_date,
            to=end_date,
            exclude_domains=ban_domains,
        )
        print(results)

        documents = []

        if "articles" not in results:
            logging.warning(
                f"No search results found for the query: {query}. Results: {results}"
            )
            cost = _PRICING["newsapi"]
            self.add_cost(cost)
            return documents, cost

        for result in results["articles"]:
            try:
                url = result["url"]
                source = result["source"]["name"]
            except KeyError:
                logging.warning(f"Invalid search result: {result}")
                continue

            documents.append(
                {
                    "url": url,
                    "favicon": None,
                    "source": source,
                    "base_url": clean_url(url),
                }
            )

            # Download the text and title of the web pages
            if download_text:
                self._download_and_process_text(documents)

                # Remove documents without text

                documents = [doc for doc in documents if "text" in doc]

        cost = _PRICING["newsapi"]
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
        start_date: datetime.datetime = None,
        end_date: datetime.datetime = None,
        search_news: bool = False,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given queries using News API.

        Args:
            queries (List[str]): The queries to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The region of the search results. Defaults to None.
            top_k (int): The number of search results to return. Defaults to 10.
            ban_domains (List[str]): A list of domains to exclude from the search results. Defaults to None.
            download_text (bool): Whether to download the text of the web pages. Defaults to True.
            start_date (datetime.datetime): The start date of the search results. Defaults to None.
            end_date (datetime.datetime): The end date of the search results. Defaults to None.
            search_news (bool): Whether to search for news articles. Defaults to False.
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
                    start_date=start_date,
                    end_date=end_date,
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
