import concurrent.futures
import datetime
import logging
from typing import Dict, List, Literal, Tuple

from serpapi import GoogleSearch

from src.api_calls.api_handler import ApiHandler, get_api_key
from src.utils.web_search import clean_url, download_text_and_title_parallel

_PRICING = {
    "serpapi": 150 / 15000,
}


class SerpAPI(ApiHandler):
    def __init__(self):
        super().__init__("serpapi")

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
        search_news: bool = False,
    ) -> Tuple[List[Dict[str, str]], float]:
        """
        Search the internet for the given query using SerpAPI.

        Args:
            query (str): The query to search for.
            language (str): The language of the search results. Defaults to "es".
            location (str): The location to search from. Defaults to None.
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

        language = language.lower()

        params = {
            "api_key": get_api_key("SerpAPI_KEY"),
            "engine": "google",
            "q": query,
            "hl": language,
            "num": top_k,
        }

        if location is not None:
            params["gl"] = location

        if start_date is not None and end_date is not None:
            params["tbs"] = (
                f"cdr:1,cd_min:{start_date.strftime('%m/%d/%Y')},cd_max:{end_date.strftime('%m/%d/%Y')}"
            )

        if search_news:
            params["tbm"] = "nws"

        search = GoogleSearch(params)
        results = search.get_dict()
        documents = []

        if "organic_results" not in results:
            logging.warning(
                f"No search results found for the query: {query}. Results: {results}"
            )
            cost = _PRICING["serpapi"]
            self.add_cost(cost)
            return documents, cost

        for result in results["organic_results"]:
            try:
                url = result["link"]
                source = result["source"]
            except KeyError:
                logging.warning(f"Invalid search result: {result}")
                continue

            if url.endswith(".pdf"):
                continue

            if "favicon" in result:
                favicon = result["favicon"]
            else:
                logging.warning(f"No favicon found for the URL: {url}")
                favicon = "https://upload.wikimedia.org/wikipedia/commons/a/a3/Image-not-found.png"

            append = True
            if ban_domains is not None:
                for domain in ban_domains:
                    if domain in url:
                        logging.warning(
                            f"Excluding URL: {url} because it contains the domain: {domain}"
                        )
                        append = False

            if append:
                documents.append(
                    {
                        "url": url,
                        "favicon": favicon,
                        "source": source,
                        "base_url": clean_url(url),
                    }
                )

            # Download the text and title of the web pages
            if download_text:
                self._download_and_process_text(documents)

                # Remove documents without text

                documents = [doc for doc in documents if "text" in doc]

        cost = _PRICING["serpapi"]
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
        Search the internet for the given queries using SerpAPI.

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

        def search_wrapper(query):
            return self.search(
                query=query,
                language=language,
                location=location,
                top_k=top_k,
                ban_domains=ban_domains,
                download_text=False,
                start_date=start_date,
                end_date=end_date,
                search_news=search_news,
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(search_wrapper, queries))

        all_documents = []
        total_cost = 0.0

        for documents, cost in results:
            all_documents.extend(documents)
            total_cost += cost

        # Deduplicate the URLs
        all_documents = list({doc["url"]: doc for doc in all_documents}.values())

        if download_text:
            self._download_and_process_text(all_documents)

        return all_documents, total_cost

    def _download_and_process_text(self, documents: List[Dict[str, str]]):
        urls_to_download = [x["url"] for x in documents]
        downloaded_data = download_text_and_title_parallel(urls=urls_to_download)

        for data, document in zip(downloaded_data, documents):
            if all(data):
                document["title"], document["text"], document["url"] = data

        documents[:] = [doc for doc in documents if "text" in doc]


"""
params = {
  "api_key": "fcb9665c0a238850735550d1d98feac9ec55bf0d91dfb8567db2bd50fe0dcc6c",
  "engine": "google",
  "q": "Elecciones parlamento vasco",
  "google_domain": "google.es",
  "tbs": "cdr:1,cd_min:1/1/2020,cd_max:21/1/2020",
  "gl": "es",
  "hl": "es",
  "tbm": "nws"
}

https://serpapi.com/playground?q=Elecciones+parlamento+vasco&google_domain=google.es&gl=es&hl=es&tbs=cdr%3A1%2Ccd_min%3A1%2F1%2F2020%2Ccd_max%3A21%2F1%2F2020&tbm=nws
"""
