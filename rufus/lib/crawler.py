"""
Crawler Module - Handles web crawling operations.

This module provides functionality to crawl websites using a BFS approach.
It extracts URLs, content, and evaluates relevance using the search and relevance check modules.
"""

import time
import queue
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from collections import deque
import concurrent.futures
from typing import List, Dict, Set, Optional, Tuple, Any

# Import logger
from rufus.helpers.logger import logger

# These will be implemented in other files
from rufus.lib.search import search_content
from rufus.lib.agents.relevance_check import RelevanceChecker
from rufus.config import config


class Crawler:
    """
    Web crawler that uses BFS approach to crawl websites.
    Extracts URLs and content, and evaluates relevance.
    """

    def __init__(
        self,
        max_depth: int = config.get("crawler.max_depth", 3),
        max_pages: int = config.get("crawler.max_pages", 100),
        timeout: int = 10,
        headers: Optional[Dict[str, str]] = None,
        respect_robots: bool = config.get("crawler.respect_robots", True),
        delay: float = config.get("crawler.crawl_delay", 0.5),
        max_workers: int = config.get("parallel.max_workers", 4),
    ):
        """
        Initialize the crawler with configurable parameters.

        Args:
            max_depth (int): Maximum depth to crawl.
            max_pages (int): Maximum number of pages to crawl.
            timeout (int): Request timeout in seconds.
            headers (Dict[str, str], optional): HTTP headers to use for requests.
            respect_robots (bool): Whether to respect robots.txt.
            delay (float): Delay between requests in seconds.
            max_workers (int): Maximum number of concurrent workers.
        """
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.timeout = timeout
        self.headers = headers or {
            "User-Agent": "Rufus-Web-Extractor/1.0 (+https://github.com/your-username/rufus)"
        }
        self.respect_robots = respect_robots
        self.delay = delay
        self.max_workers = max_workers

        # Visited URLs
        self.visited_urls: Set[str] = set()

        # Results storage
        self.extracted_content: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"Crawler initialized with max_depth={max_depth}, max_pages={max_pages}"
        )

    def _is_valid_url(self, url: str, base_url: str) -> bool:
        """
        Check if a URL is valid for crawling.

        Args:
            url (str): URL to check.
            base_url (str): Base URL for comparison.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        parsed_url = urlparse(url)
        parsed_base = urlparse(base_url)

        # Check if the URL is from the same domain
        if parsed_url.netloc and parsed_url.netloc != parsed_base.netloc:
            return False

        # Skip anchors, javascript, mailto links, etc.
        if not parsed_url.netloc and not parsed_url.path:
            return False

        # Skip certain file types (images, videos, documents, etc.)
        skip_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".mp3",
            ".mp4",
            ".zip",
            ".tar",
            ".gz",
        ]
        if any(parsed_url.path.endswith(ext) for ext in skip_extensions):
            return False

        return True

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract links from HTML content.

        Args:
            soup (BeautifulSoup): BeautifulSoup object of the page.
            base_url (str): Base URL of the page.

        Returns:
            List[str]: List of normalized URLs.
        """
        links = []

        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "")
            normalized_url = urljoin(base_url, href)

            if self._is_valid_url(normalized_url, base_url):
                links.append(normalized_url)

        logger.debug(f"Extracted {len(links)} links from {base_url}")
        return links

    def _extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract relevant content from HTML.

        Args:
            soup (BeautifulSoup): BeautifulSoup object of the page.
            url (str): URL of the page.

        Returns:
            Dict[str, Any]: Extracted content with metadata.
        """
        # Extract title
        title = soup.title.string if soup.title else "No Title"

        # Extract main content - this is a simple approach
        # In practice, we'd need more sophisticated content extraction
        content = ""
        main_content_tags = [
            "article",
            "main",
            'div[role="main"]',
            ".content",
            "#content",
        ]

        # Try to find main content using common patterns
        for tag in main_content_tags:
            main_content = soup.select(tag)
            if main_content:
                content = main_content[0].get_text(strip=True, separator=" ")
                break

        # If no main content found, use the body
        if not content and soup.body:
            # Extract text from body, excluding script, style, etc.
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            content = soup.body.get_text(strip=True, separator=" ")

        # Extract headings for structure
        headings = []
        for i in range(1, 7):
            for h in soup.find_all(f"h{i}"):
                headings.append(f"H{i}: {h.get_text(strip=True)}")

        # Extract meta description
        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag:
            meta_desc = meta_tag.get("content", "")

        # Create content object
        content_obj = {
            "url": url,
            "title": title,
            "meta_description": meta_desc,
            "headings": headings,
            "content": content,
            "extracted_at": time.time(),
        }

        logger.debug(f"Extracted content from {url}, content length: {len(content)}")
        return content_obj

    def _check_page_relevance(self, content: Dict[str, Any], main_content: str) -> bool:
        """
        Check if a page is relevant compared to the main content.
        Uses the relevance check module with LangChain LLM agent.

        Args:
            content (Dict[str, Any]): Extracted content of current page.
            main_content (str): Content from the main/starting page or user instructions.

        Returns:
            bool: True if page is relevant, False otherwise.
        """
        try:
            # Prepare content for relevance check
            text_to_check = f"Title: {content['title']}\n"
            if content["meta_description"]:
                text_to_check += f"Description: {content['meta_description']}\n"

            # Add some headings if available
            if content["headings"]:
                headings_text = "\n".join(content["headings"][:5])  # First 5 headings
                text_to_check += f"Headings: {headings_text}\n"

            # Add a snippet of content
            content_snippet = (
                content["content"][:500]
                if len(content["content"]) > 500
                else content["content"]
            )
            text_to_check += f"Content snippet: {content_snippet}"

            # Call the relevance check function using LangChain LLM agent
            is_relevant = RelevanceChecker.check_relevance(
                main_content=main_content, check_content=text_to_check
            )

            if is_relevant:
                logger.info(f"Page {content['url']} is relevant to the main content")
            else:
                logger.info(
                    f"Page {content['url']} is not relevant to the main content"
                )

            return is_relevant

        except Exception as e:
            logger.error(
                f"Error checking relevance for {content['url']}: {str(e)}",
                exc_info=True,
            )
            # Default to True in case of error to include potentially relevant content
            return True

    def _fetch_page(self, url: str) -> Tuple[Optional[str], Optional[BeautifulSoup]]:
        """
        Fetch a page and return its content.

        Args:
            url (str): URL to fetch.

        Returns:
            Tuple[Optional[str], Optional[BeautifulSoup]]:
                Tuple of (normalized URL, BeautifulSoup object) or (None, None) on error.
        """
        try:
            print(
                f"\rProcessed URLs: {len(self.visited_urls)}/{self.max_pages} | Current: {url}",
                end="",
                flush=True,
            )
            logger.info(f"Fetching page: {url}")
            response = requests.get(url, headers=self.headers, timeout=self.timeout)

            # Check if the request was successful
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
                return None, None

            # Get the normalized URL (handle redirects)
            normalized_url = response.url

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            return normalized_url, soup

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None, None

    def crawl(self, start_url: str, instructions: str = "") -> Dict[str, Any]:
        """
        Crawl a website starting from the given URL using BFS approach.

        Args:
            start_url (str): URL to start crawling from.
            instructions (str): User instructions for relevance checking.

        Returns:
            Dict[str, Any]: Dictionary containing relevant_pages, all_visited_urls, and queue_links.
        """
        logger.info(
            f"Starting crawl from {start_url} with instructions: {instructions}"
        )

        # Create tmp directory if it doesn't exist
        import os
        import json
        import csv
        from datetime import datetime

        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            logger.info(f"Created tmp directory at {tmp_dir}")

        # Reset state for new crawl
        self.visited_urls = set()
        self.extracted_content = {}
        queue_links = []  # To track links in the queue

        # BFS queue with (url, depth) tuples
        url_queue = deque([(start_url, 0)])
        pages_processed = 0

        # Get main content from the start URL to use as reference for relevance checks
        normalized_start_url, start_soup = self._fetch_page(start_url)
        if not start_soup:
            logger.error(f"Failed to fetch start URL: {start_url}")
            return {
                "relevant_pages": {},
                "all_visited_urls": list(self.visited_urls),
                "queue_links": queue_links,
            }

        main_content = self._extract_content(start_soup, normalized_start_url)
        main_content_text = f"Title: {main_content['title']}\n"
        if main_content["meta_description"]:
            main_content_text += f"Description: {main_content['meta_description']}\n"
        main_content_text += f"Content: {main_content['content'][:1000]}"

        # If instructions are provided, use them to augment the main content
        if instructions:
            main_content_text = f"{instructions}\n\n{main_content_text}"

        # Store the start page content
        self.extracted_content[normalized_start_url] = main_content
        pages_processed += 1

        # Extract links from start page and add to queue
        start_links = self._extract_links(start_soup, normalized_start_url)
        for link in start_links:
            if link not in self.visited_urls:
                url_queue.append((link, 1))
                queue_links.append(link)

        while url_queue and pages_processed < self.max_pages:
            # Get the next URL and its depth
            current_url, current_depth = url_queue.popleft()

            # Remove from queue_links as we're processing it now
            if current_url in queue_links:
                queue_links.remove(current_url)

            # Skip if already visited
            if current_url in self.visited_urls:
                continue

            # Mark as visited
            self.visited_urls.add(current_url)

            # Skip start URL as we've already processed it
            if current_url == start_url or current_url == normalized_start_url:
                continue

            # Fetch the page
            normalized_url, soup = self._fetch_page(current_url)

            # Skip if fetch failed
            if not soup:
                continue

            # Extract basic content for relevance checking
            content = self._extract_content(soup, normalized_url)

            # Check if the content is relevant using the main content as reference
            if not self._check_page_relevance(content, main_content_text):
                logger.info(f"Skipping irrelevant page: {normalized_url}")
                continue

            # Page is relevant, use search_content to extract the actual content we want
            extracted_content = self._extract_page_content(normalized_url)
            if extracted_content:
                # Merge with basic content data
                content.update(extracted_content)

            # Store the content
            self.extracted_content[normalized_url] = content
            pages_processed += 1

            logger.info(
                f"Processed page {pages_processed}/{self.max_pages}: {normalized_url}"
            )

            # If we've reached the maximum depth, don't extract more links
            if current_depth >= self.max_depth:
                continue

            # Extract links and add them to the queue
            links = self._extract_links(soup, normalized_url)
            for link in links:
                if link not in self.visited_urls and link not in queue_links:
                    url_queue.append((link, current_depth + 1))
                    queue_links.append(link)

            # Respect the delay between requests
            time.sleep(self.delay)

        logger.info(
            f"Crawl completed. Visited {len(self.visited_urls)} URLs, extracted content from {len(self.extracted_content)} pages"
        )
        print(f"saving crawl results with {len(self.extracted_content)} relevant pages")

        # Save results to files in tmp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save relevant pages as JSON
        json_file = os.path.join(tmp_dir, f"relevant_pages_{timestamp}.json")
        with open(json_file, "w") as f:
            json.dump(self.extracted_content, f, indent=2)
        logger.info(f"Saved relevant pages to {json_file}")

        # Save all visited URLs as CSV
        csv_file = os.path.join(tmp_dir, f"visited_urls_{timestamp}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["URL"])
            for url in self.visited_urls:
                writer.writerow([url])
        logger.info(f"Saved visited URLs to {csv_file}")

        # Save queue links as CSV
        queue_csv_file = os.path.join(tmp_dir, f"queue_links_{timestamp}.csv")
        with open(queue_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["URL"])
            for url in queue_links:
                writer.writerow([url])
        logger.info(f"Saved queue links to {queue_csv_file}")

        print(
            f"Returning crawl results with {len(self.extracted_content)} relevant pages"
        )

        # Return the crawl results
        return {
            "relevant_pages": self.extracted_content,
            "all_visited_urls": list(self.visited_urls),
            "queue_links": queue_links,
        }

    def _extract_page_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a page using the search module.
        This uses search_content to get the actual content we want from the page.

        Args:
            url (str): URL to extract content from.

        Returns:
            Dict[str, Any]: Extracted content or empty dict on failure.
        """
        logger.info(f"Extracting content from {url} using search_content")

        try:
            # Call the search_content function to extract content from the page
            content = search_content(url)

            if content:
                logger.info(f"Successfully extracted content from {url}")
                return content
            else:
                logger.warning(f"No content extracted from {url}")
                return {}

        except Exception as e:
            logger.error(
                f"Error extracting content from {url}: {str(e)}", exc_info=True
            )
            return {}

    def crawl_parallel(self, start_url: str, instructions: str = "") -> Dict[str, Any]:
        """
        Crawl a website in parallel using a thread pool.
        This is an advanced version of the crawl method.

        Args:
            start_url (str): URL to start crawling from.
            instructions (str): User instructions for relevance checking.

        Returns:
            Dict[str, Any]: Dictionary containing relevant_pages, all_visited_urls, and queue_links.
        """
        logger.info(f"Starting parallel crawl from {start_url}")

        # Create tmp directory if it doesn't exist
        import os
        import json
        import csv
        from datetime import datetime

        tmp_dir = "tmp"
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
            logger.info(f"Created tmp directory at {tmp_dir}")

        # Reset state for new crawl
        self.visited_urls = set()
        self.extracted_content = {}
        queue_links = []  # To track links in the queue

        # Thread synchronization
        from threading import Lock

        lock = Lock()

        # Get main content from the start URL to use as reference for relevance checks
        normalized_start_url, start_soup = self._fetch_page(start_url)
        if not start_soup:
            logger.error(f"Failed to fetch start URL: {start_url}")
            return {
                "relevant_pages": {},
                "all_visited_urls": list(self.visited_urls),
                "queue_links": queue_links,
            }

        main_content = self._extract_content(start_soup, normalized_start_url)
        main_content_text = f"Title: {main_content['title']}\n"
        if main_content["meta_description"]:
            main_content_text += f"Description: {main_content['meta_description']}\n"
        main_content_text += f"Content: {main_content['content'][:1000]}"

        # If instructions are provided, use them to augment the main content
        if instructions:
            main_content_text = f"{instructions}\n\n{main_content_text}"

        # Store the start page content
        with lock:
            self.extracted_content[normalized_start_url] = main_content
            pages_processed = 1

        # Use a regular Queue for thread safety
        url_queue = queue.Queue()

        # Extract links from start page and add to queue
        start_links = self._extract_links(start_soup, normalized_start_url)
        for link in start_links:
            url_queue.put((link, 1))  # Start with depth 1 for links from the main page
            with lock:
                queue_links.append(link)

        stop_processing = False

        def process_url():
            nonlocal pages_processed, stop_processing

            while (
                not url_queue.empty()
                and pages_processed < self.max_pages
                and not stop_processing
            ):
                try:
                    # Get the next URL and its depth
                    current_url, current_depth = url_queue.get(block=False)

                    # Remove from queue_links as we're processing it
                    with lock:
                        if current_url in queue_links:
                            queue_links.remove(current_url)

                    # Skip if already visited
                    with lock:
                        if current_url in self.visited_urls:
                            url_queue.task_done()
                            continue

                        # Mark as visited
                        self.visited_urls.add(current_url)

                    # Fetch the page
                    normalized_url, soup = self._fetch_page(current_url)

                    # Skip if fetch failed
                    if not soup:
                        url_queue.task_done()
                        continue

                    # Extract basic content for relevance checking
                    content = self._extract_content(soup, normalized_url)

                    # Check if the content is relevant using the main content as reference
                    is_relevant = self._check_page_relevance(content, main_content_text)

                    if is_relevant:
                        # Page is relevant, use search_content to extract the actual content we want
                        extracted_content = self._extract_page_content(normalized_url)
                        if extracted_content:
                            # Merge with basic content data
                            content.update(extracted_content)

                        # Store the content
                        with lock:
                            self.extracted_content[normalized_url] = content
                            pages_processed += 1

                        if pages_processed >= self.max_pages:
                            print(f"\nReached max pages limit: {self.max_pages}")
                            # Clear remaining queue to unblock join()
                            try:
                                while not url_queue.empty():
                                    url_queue.get(block=False)
                                    url_queue.task_done()
                            except queue.Empty:
                                pass
                            url_queue.task_done()
                            stop_processing = True
                            break

                        logger.info(
                            f"Processed page {pages_processed}/{self.max_pages}: {normalized_url}"
                        )

                        # If we've reached the maximum depth, don't extract more links
                        if current_depth < self.max_depth:
                            # Extract links and add them to the queue
                            links = self._extract_links(soup, normalized_url)
                            with lock:
                                for link in links:
                                    if (
                                        link not in self.visited_urls
                                        and link not in queue_links
                                    ):
                                        url_queue.put((link, current_depth + 1))
                                        queue_links.append(link)
                    else:
                        logger.info(f"Skipping irrelevant page: {normalized_url}")
                        print(f"<<< Skipping irrelevant page: {normalized_url} >>>")

                    # Respect the delay between requests
                    time.sleep(self.delay)

                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing URL: {str(e)}", exc_info=True)
                finally:
                    url_queue.task_done()

        print(f"Starting parallel crawl with {self.max_workers} workers...")

        # Create and start worker threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            # Start workers
            workers = [executor.submit(process_url) for _ in range(self.max_workers)]

            # Wait for all tasks to be processed
            try:
                print("Waiting for queue to complete...")
                # Add a timeout to join to avoid hanging
                url_queue.join(timeout=10)  # 10 second timeout
                print("Queue processed successfully")
            except Exception as e:
                print(
                    f"Queue join completed or timed out: {str(e) if not isinstance(e, TimeoutError) else 'Timeout'}"
                )

            # Cancel any remaining tasks
            for worker in workers:
                worker.cancel()

        logger.info(
            f"Parallel crawl completed. Visited {len(self.visited_urls)} URLs, extracted content from {len(self.extracted_content)} pages"
        )
        print(
            f"Saving parallel crawl results with {len(self.extracted_content)} relevant pages"
        )

        # Save results to files in tmp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save relevant pages as JSON
        json_file = os.path.join(tmp_dir, f"relevant_pages_{timestamp}.json")
        with open(json_file, "w") as f:
            json.dump(self.extracted_content, f, indent=2)
        logger.info(f"Saved relevant pages to {json_file}")

        # Save all visited URLs as CSV
        csv_file = os.path.join(tmp_dir, f"visited_urls_{timestamp}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["URL"])
            for url in self.visited_urls:
                writer.writerow([url])
        logger.info(f"Saved visited URLs to {csv_file}")

        # Save queue links as CSV
        queue_csv_file = os.path.join(tmp_dir, f"queue_links_{timestamp}.csv")
        with open(queue_csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["URL"])
            for url in queue_links:
                writer.writerow([url])
        logger.info(f"Saved queue links to {queue_csv_file}")

        # Return the crawl results
        return {
            "relevant_pages": self.extracted_content,
            "all_visited_urls": list(self.visited_urls),
            "queue_links": queue_links,
        }


# Simple function to crawl a website
def crawl_website(url: str, instructions: str = "", **kwargs) -> Dict[str, Any]:
    """
    Crawl a website and return extracted content and crawl information.

    Args:
        url (str): URL to crawl.
        instructions (str): User instructions for relevance checking and content extraction.
        **kwargs: Additional arguments for the Crawler.

    Returns:
        Dict[str, Any]: Dictionary containing relevant_pages, all_visited_urls, and queue_links.
    """
    logger.info(f"Starting website crawl for {url} with instructions: {instructions}")
    crawler = Crawler(**kwargs)
    return crawler.crawl(url, instructions)


# Parallel crawling function
def crawl_website_parallel(
    url: str, instructions: str = "", **kwargs
) -> Dict[str, Any]:
    """
    Crawl a website in parallel and return extracted content and crawl information.

    Args:
        url (str): URL to crawl.
        instructions (str): User instructions for relevance checking and content extraction.
        **kwargs: Additional arguments for the Crawler.

    Returns:
        Dict[str, Any]: Dictionary containing relevant_pages, all_visited_urls, and queue_links.
    """
    logger.info(
        f"Starting parallel website crawl for {url} with instructions: {instructions}"
    )
    crawler = Crawler(**kwargs)
    return crawler.crawl_parallel(url, instructions)


# Demo/testing code
if __name__ == "__main__":
    # Test the crawler with python.org
    test_url = "https://www.python.org"
    test_instructions = (
        "Information about Python programming language, features, and documentation"
    )

    print(f"Crawling {test_url}...")
    results = crawl_website(
        test_url, test_instructions, max_depth=2, max_pages=5, delay=1.0
    )

    print(f"Crawled {len(results['relevant_pages'])} relevant pages:")
    for url, content in results["relevant_pages"].items():
        print(f"- {url}: {content['title']}")

    print(f"\nTotal visited URLs: {len(results['all_visited_urls'])}")
    print(f"URLs still in queue: {len(results['queue_links'])}")
