"""
RufusClient - Python client for the Rufus Web Extractor.

This module provides a simple client interface for interacting with Rufus,
matching the expected API pattern from the case study.
"""

import os
import requests
import json
from typing import Dict, List, Any, Optional, Union
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RufusClient:
    """
    Client for interacting with the Rufus Web Extractor.
    Provides methods for scraping websites and querying extracted data.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8234",
        model: str = "ollama",
    ):
        """
        Initialize the Rufus client.

        Args:
            api_key: API key for authentication (optional)
            base_url: Base URL of the Rufus API
            model: LLM model to use ('ollama' or 'openai')
        """
        self.model = model.lower()

        # Determine which API key to use
        if api_key is None:
            if self.model == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OpenAI API key not found. Please provide it or set OPENAI_API_KEY in .env file."
                    )
            elif self.model == "ollama":
                # Ollama typically doesn't require an API key, but we'll check for a custom one
                api_key = os.getenv("OLLAMA_API_KEY")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

        # Add API key to headers if provided
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

        # Add model info to headers
        self.headers["X-Model-Type"] = self.model

    def scrape(
        self,
        url: str,
        instructions: str = "",
        max_depth: int = 2,
        max_pages: int = 20,
        use_parallel: bool = True,
    ) -> Dict[str, Any]:
        """
        Scrape a website and prepare data for RAG.

        Args:
            url: URL to scrape
            instructions: User instructions for content extraction
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            use_parallel: Whether to use parallel crawling

        Returns:
            Dictionary containing processed documents and metadata
        """
        # Prepare request payload
        payload = {
            "url": url,
            "instructions": instructions
            or f"Extract comprehensive information from {url}",
            "max_depth": max_depth,
            "max_pages": max_pages,
            "model": self.model,
        }

        # Send scrape request
        try:
            response = requests.post(
                f"{self.base_url}/scrape",
                headers=self.headers,
                json=payload,
                timeout=300,  # 2-minute timeout for initial request
            )

            # Check response
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", "Unknown error")
                except:
                    error_msg = f"HTTP error {response.status_code}"

                raise Exception(f"Failed to scrape website: {error_msg}")
        except requests.exceptions.Timeout:
            raise Exception(
                "Request timed out. The server might be busy or unavailable."
            )
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Connection error. Please make sure the API server is running at {self.base_url}"
            )

        # Get job details
        job_data = response.json()
        job_id = job_data.get("job_id")

        print(f"Job created: {job_id}")
        print("Processing website. This may take a few minutes...")

        # Poll for job completion
        max_retries = 60  # Maximum number of retries (2 minutes with 2-second delay)
        retries = 0

        while job_data.get("status") in ["pending", "processing"]:
            time.sleep(2)  # Wait for 2 seconds before checking again

            try:
                job_response = requests.get(
                    f"{self.base_url}/jobs/{job_id}", headers=self.headers, timeout=10
                )

                if job_response.status_code != 200:
                    raise Exception("Failed to retrieve job status")

                job_data = job_response.json()

                # Print progress update every 5 retries
                if retries % 5 == 0:
                    status = job_data.get("status", "unknown")
                    print(f"Current status: {status}... (waiting)")

                # Break if job is completed or failed
                if job_data.get("status") in ["completed", "failed"]:
                    break

                retries += 1
                if retries >= max_retries:
                    raise Exception(
                        "Job processing timed out. Please check the server logs."
                    )

            except requests.exceptions.RequestException as e:
                raise Exception(f"Error checking job status: {str(e)}")

        # Check if job failed
        if job_data.get("status") == "failed":
            raise Exception(f"Job failed: {job_data.get('error', 'Unknown error')}")

        # If completed, return documents
        collection_name = job_data.get("collection_name")
        print(f"Job completed. Collection name: {collection_name}")

        # Return a documented data structure
        return {
            "job_id": job_id,
            "collection_name": collection_name,
            "url": url,
            "status": job_data.get("status"),
            "completed_at": job_data.get("completed_at"),
            "model": self.model,
        }

    def query(
        self, collection_name: str, query: str, custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query a processed collection using RAG.

        Args:
            collection_name: Name of the collection to query
            query: Query text
            custom_prompt: Optional custom prompt for the LLM

        Returns:
            Dictionary containing answer and source information
        """
        # Prepare request payload
        payload = {
            "collection_name": collection_name,
            "query": query,
            "model": self.model,
        }

        if custom_prompt:
            payload["custom_prompt"] = custom_prompt

        # Send query request
        try:
            response = requests.post(
                f"{self.base_url}/query", headers=self.headers, json=payload, timeout=30
            )

            # Check response
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("detail", "Unknown error")
                except:
                    error_msg = f"HTTP error {response.status_code}"

                raise Exception(f"Failed to query collection: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {str(e)}")

        # Return query results
        return response.json()

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all available collections.

        Returns:
            List of collection information
        """
        try:
            response = requests.get(
                f"{self.base_url}/collections", headers=self.headers, timeout=10
            )

            if response.status_code != 200:
                raise Exception("Failed to list collections")

            return response.json().get("collections", [])
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {str(e)}")

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            True if deletion was successful
        """
        try:
            response = requests.delete(
                f"{self.base_url}/collections/{collection_name}",
                headers=self.headers,
                timeout=10,
            )

            if response.status_code != 200:
                raise Exception(f"Failed to delete collection: {collection_name}")

            return True
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request error: {str(e)}")

    def get_health(self) -> Dict[str, Any]:
        """
        Check the health of the API service.

        Returns:
            Dictionary containing health status information
        """
        try:
            response = requests.get(
                f"{self.base_url}/health", headers=self.headers, timeout=5
            )

            if response.status_code != 200:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}

            return response.json()
        except requests.exceptions.RequestException as e:
            return {"status": "unhealthy", "error": str(e)}


# # Example usage
# if __name__ == "__main__":
#     # Create client with ollama model (default)
#     client_ollama = RufusClient()

#     # Or create client with OpenAI
#     # client_openai = RufusClient(model="openai")

#     try:
#         # Check API health
#         health = client_ollama.get_health()
#         if health.get("status") != "healthy":
#             print(f"Warning: API service appears to be unhealthy: {health.get('error', 'Unknown error')}")
#             print("Continuing anyway...")

#         # Scrape a website
#         print("Scraping website...")
#         result = client_ollama.scrape(
#             url="https://www.python.org",
#             instructions="Extract information about Python features and documentation",
#             max_depth=1,
#             max_pages=5
#         )

#         print(f"Scraping completed: {result}")

#         # Query the collection
#         if "collection_name" in result:
#             print("\nQuerying collection...")
#             answer = client_ollama.query(
#                 collection_name=result["collection_name"],
#                 query="What are the main features of Python?"
#             )

#             print(f"Answer: {answer.get('answer')}")
#             print("\nSources:")
#             for url in answer.get("source_urls", []):
#                 print(f"- {url}")

#     except Exception as e:
#         print(f"Error: {str(e)}")

# At the end of client.py
if __name__ == "__main__":
    # Ask user if they want to run the example
    run_example = input(
        "Do you want to run the example scraping and query? (y/n): "
    ).lower()

    if run_example == "y":
        # Create a client with ollama model (default)
        client_ollama = RufusClient()

        try:
            # Optional: Prompt user for URL
            url = (
                input("Enter URL to scrape (default: https://www.python.org): ")
                or "https://www.python.org"
            )
            instructions = (
                input(
                    "Enter instructions for scraping (default: Extract information about Python programming language and documentation): "
                )
                or "Extract information about Python programming language and documentation"
            )
            max_depth = int(input("Enter maximum crawl depth (default: 1): ") or 1)
            max_pages = int(input("Enter maximum pages to crawl (default: 5): ") or 5)

            # Scrape website
            print(f"Scraping {url}...")
            result = client_ollama.scrape(
                url=url,
                instructions=instructions,
                max_depth=max_depth,
                max_pages=max_pages,
            )

            print(f"Scraping completed: {result}")

            # Optional: Prompt user for query
            query = (
                input(
                    "Enter a query about the website (default: What are the main features of Python?): "
                )
                or "What are the main features of Python?"
            )

            # Query the collection
            if "collection_name" in result:
                print("\nQuerying collection...")
                answer = client_ollama.query(
                    collection_name=result["collection_name"], query=query
                )

                print(f"Answer: {answer.get('answer')}")
                print("\nSources:")
                for url in answer.get("source_urls", []):
                    print(f"- {url}")

        except Exception as e:
            print(f"Error: {str(e)}")
    else:
        print("Example execution skipped.")
