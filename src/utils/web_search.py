import asyncio
import multiprocessing
import re
import threading
from functools import lru_cache
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup


def run_async_in_thread(coroutine):
    result = []
    exception = []

    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result.append(loop.run_until_complete(coroutine))
        except Exception as e:
            exception.append(e)
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()

    if exception:
        raise exception[0]
    return result[0]


def is_valid_url(url):
    """Check if a string is a valid URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def clean_url(full_url):
    """
    Clean a URL by removing any query parameters.

    Args:
        full_url (str): The full URL.

    Returns:
        str: The cleaned URL.
    """
    # Remove special characters
    full_url = full_url.encode("ascii", "ignore").decode("ascii")
    # Remove query parameters
    parsed_url = urlparse(full_url)
    # Build the base URL
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

    return base_url


def get_domain_name(url: str) -> str:
    """
    Get the domain name from a URL.

    Args:
        url (str): The URL.

    Returns:
        str: The domain name.
    """

    parsed_uri = urlparse(url)
    return "{uri.netloc}".format(uri=parsed_uri)


@lru_cache(maxsize=128)
async def get_favicon_async(
    url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> str:
    """
    Download the favicon for a given URL asynchronously.
    """
    default_favicon = (
        "https://upload.wikimedia.org/wikipedia/commons/0/01/Website_icon.svg"
    )

    async with semaphore:
        try:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, "lxml")

                    icon_links = soup.find_all(
                        "link",
                        rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                    )
                    meta_icons = soup.find_all(
                        "meta", attrs={"content": re.compile(r".ico$", re.I)}
                    )
                    icons = icon_links + meta_icons

                    for icon in icons:
                        favicon_url = icon.get("href") or icon.get("content")
                        if favicon_url:
                            if favicon_url.startswith("/"):
                                favicon_url = urljoin(url, favicon_url)
                            return favicon_url

                    return default_favicon
        except Exception as e:
            print(f"Error fetching favicon for {url}: {e}")

    return default_favicon


async def download_favicons_async(urls: List[str]) -> Dict[str, str]:
    """
    Download the favicons for a list of URLs asynchronously.
    """
    favicons = {}
    unique_urls = list(set(urls))
    num_connections = min(100, max(2, multiprocessing.cpu_count() * 5))
    semaphore = asyncio.Semaphore(num_connections)

    async with aiohttp.ClientSession(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
    ) as session:
        tasks = [get_favicon_async(url, session, semaphore) for url in unique_urls]
        results = await asyncio.gather(*tasks)

        for url, favicon_url in zip(unique_urls, results):
            favicons[url] = favicon_url

    return favicons


def download_favicons(urls: List[str]) -> Dict[str, str]:
    """
    Download the favicons for a list of URLs.

    Args:
        urls (List[str]): The list of URLs.

    Returns:
        Dict[str, str]: A dictionary mapping URLs to their favicons.
    """
    return run_async_in_thread(download_text_and_title_parallel_async(urls))

    # return asyncio.run(download_favicons_async(urls))


async def get_final_url_and_soup(
    url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> Tuple[str, BeautifulSoup]:
    """
    Follow redirects and return the final URL and BeautifulSoup object.
    """
    async with semaphore:
        try:
            async with session.get(url, allow_redirects=True, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    return str(response.url), BeautifulSoup(content, "html.parser")
        except Exception as e:
            print(f"Request failed for {url}: {e}")
    return None, None


def create_session():
    """Create a requests session for connection pooling."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        }
    )
    return session


def clean_text(text):
    """
    Clean the text by removing extra whitespace and newlines.

    Args:
        text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = " ".join(text.strip().split()).strip()
    return text


async def download_text_and_title_async(
    url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> Tuple[str, str, str]:
    """
    Download title and text from a URL, handling redirects and title-URLs.
    """
    initial_url = url.encode("ascii", "ignore").decode("ascii")
    final_url, soup = await get_final_url_and_soup(initial_url, session, semaphore)

    if soup:
        title = soup.title.string.strip() if soup.title else "No Title Found"

        if is_valid_url(title) and title != final_url:
            final_url, soup = await get_final_url_and_soup(title, session, semaphore)
            title = soup.title.string.strip() if soup.title else "No Title Found"
            if not soup:
                return None, None, None

        text_elements = [clean_text(p.text.strip()) for p in soup.select("p")]
        text = "\n".join(filter(None, text_elements))

        return title, text, final_url

    return None, None, None


async def download_text_and_title_parallel_async(
    urls: List[str],
) -> List[Tuple[str, str, str]]:
    """
    Download title and text from a list of URLs in parallel using asyncio.

    Args:
        urls (List[str]): The list of URLs to download.

    Returns:
        List[Tuple[str, str, str]]: The list of title, text, and final URL tuples.
    """
    num_connections = min(100, max(2, multiprocessing.cpu_count() * 5))
    semaphore = asyncio.Semaphore(num_connections)

    async with aiohttp.ClientSession(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
    ) as session:
        tasks = [download_text_and_title_async(url, session, semaphore) for url in urls]
        results = await asyncio.gather(*tasks)

    return results


def download_text_and_title_parallel(urls: List[str]) -> List[Tuple[str, str, str]]:
    """
    Wrapper function to run the async function in both synchronous and asynchronous contexts.
    """

    return run_async_in_thread(download_text_and_title_parallel_async(urls))

    # try:
    # Try running in an existing event loop
    #    loop = asyncio.get_event_loop()
    #    if loop.is_running():
    #        return loop.run_until_complete(download_text_and_title_parallel_async(urls))
    #    else:
    #        return asyncio.run(download_text_and_title_parallel_async(urls))
    # except RuntimeError:
    # If there's no event loop, create a new one
    #    return asyncio.run(download_text_and_title_parallel_async(urls))


async def download_text_title_favicon_async(
    url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore
) -> Tuple[str, str, str, str]:
    """
    Download title, text, final URL, and favicon from a URL asynchronously.
    """
    async with semaphore:
        try:
            async with session.get(url, allow_redirects=True, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    final_url = str(response.url)
                    soup = BeautifulSoup(content, "html.parser")

                    # Extract title and text
                    title = (
                        soup.title.string.strip() if soup.title else "No Title Found"
                    )
                    text_elements = [
                        clean_text(p.text.strip()) for p in soup.select("p")
                    ]
                    text = "\n".join(filter(None, text_elements))

                    # Extract favicon
                    favicon_url = "https://upload.wikimedia.org/wikipedia/commons/0/01/Website_icon.svg"
                    icon_links = soup.find_all(
                        "link",
                        rel=re.compile(r"(shortcut icon|icon|apple-touch-icon)", re.I),
                    )
                    meta_icons = soup.find_all(
                        "meta", attrs={"content": re.compile(r".ico$", re.I)}
                    )
                    icons = icon_links + meta_icons

                    for icon in icons:
                        favicon_url = icon.get("href") or icon.get("content")
                        if favicon_url:
                            if favicon_url.startswith("/"):
                                favicon_url = urljoin(final_url, favicon_url)
                            break

                    return title, text, final_url, favicon_url
        except Exception as e:
            print(f"Error processing {url}: {e}")

    return None, None, None, None


async def download_text_favicon_parallel_async(
    urls: List[str],
) -> List[Tuple[str, str, str, str]]:
    """
    Download title, text, final URL, and favicon from a list of URLs in parallel using asyncio.
    """
    num_connections = min(100, max(2, multiprocessing.cpu_count() * 5))
    semaphore = asyncio.Semaphore(num_connections)

    async with aiohttp.ClientSession(
        #connector=aiohttp.TCPConnector(ssl=False), 
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
    ) as session:
        tasks = {
            url: asyncio.create_task(
                download_text_title_favicon_async(url, session, semaphore)
            )
            for url in urls
        }

        # Wait for all tasks to complete
        await asyncio.gather(*tasks.values())

        # Reorder results based on input URL order
        results = []
        for url in urls:
            result = await tasks[url]
            results.append(result)

    return results


def download_text_favicon_parallel(
    urls: List[str],
) -> List[Tuple[str, str, str, str]]:
    """
    Wrapper function to run the async function in both synchronous and asynchronous contexts.
    """
    return run_async_in_thread(download_text_favicon_parallel_async(urls))


if __name__ == "__main__":
    import time

    urls = [
        "https://elchapuzasinformatico.com/2024/08/coolpc-gamer-xi-powered-by-msi/",
        "https://elchapuzasinformatico.com/2024/08/gafas-xr-samsung-soc-snapdragon-xr2-gen-2/",
        "https://www.elcorreo.com/sociedad/educacion/red-publica-25000-profesores-doble-concertada-similares-20240828010340-nt.html",
        "https://elpais.com/economia/2024-08-28/el-cni-alerto-de-la-conexion-rusa-del-grupo-hungaro-que-puja-por-talgo.html",
        "https://www.elmundo.es/espana/2024/08/27/66ce074be4d4d80d358b4582.html",
        "https://www.marca.com/futbol/barcelona/2024/08/28/66ce528d22601d0c488b4583.html",
        "https://www.sport.es/es/noticias/barca/promesa-nico-williams-le-aleja-106197709",
    ] * 8

    start_time = time.time()
    results = download_text_and_title_parallel(urls)
    total_time_text = time.time() - start_time
    for title, text, url in results[:3]:
        print(f"Title: {title}")
        print(f"URL: {url}")
        print(f"Text: {text[:100]}...\n")

    print("____________________________________________________")

    # Test favicon download
    start_time = time.time()
    favicons = download_favicons(urls)
    total_time_fav = time.time() - start_time

    # Test combined download

    start_time = time.time()
    results = download_text_favicon_parallel(urls)
    total_time_combined = time.time() - start_time
    for title, text, url, favicon_url in results[:3]:
        print(f"Title: {title}")
        print(f"URL: {url}")
        print(f"Favicon: {favicon_url}")
        print(f"Text: {text[:100]}...\n")

    print(f"Total time text: {total_time_text:.2f} seconds")
    print(f"Downloaded {len(favicons)} favicons in {total_time_fav:.2f} seconds")
    print(f"Total time combined: {total_time_combined:.2f} seconds")
