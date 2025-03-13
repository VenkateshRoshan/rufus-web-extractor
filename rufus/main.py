"""
Main module for Rufus Web Extractor.
Orchestrates the web crawling, content extraction, and RAG pipeline.
Updated with support for multiple LLM models.
"""

from typing import Dict, List, Any, Optional
import os
import json
from datetime import datetime

# Import Rufus components
from rufus.lib.crawler import crawl_website, crawl_website_parallel
from rufus.lib.agents.relevance_check import RelevanceChecker
from rufus.lib.rag.vectordb_creation import VectorDBCreator
from rufus.lib.rag.rag import RAGHandler
from rufus.helpers.logger import logger
from rufus.config import config

# Import for environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RufusExtractor:
    """Main class for orchestrating the Rufus web extraction pipeline."""

    def __init__(
        self,
        use_parallel: bool = config.get("parallel.enabled", True),
        max_depth: int = config.get("crawler.max_depth", 3),
        max_pages: int = config.get("crawler.max_pages", 100),
        relevance_threshold: float = config.get("relevance.threshold", 0.5),
        model: str = config.get("model.name", "ollama"),
        max_workers: int = config.get("parallel.max_workers", 4),  # Added max_workers
    ):
        """
        Initialize the Rufus extraction pipeline.

        Args:
            use_parallel: Whether to use parallel crawling
            max_depth: Maximum crawl depth
            max_pages: Maximum pages to crawl
            relevance_threshold: Minimum relevance score for content
            model: LLM model to use ('ollama' or 'openai')
        """
        # Rest of the initialization remains the same
        self.use_parallel = use_parallel
        self.crawler_config = {
            "max_depth": max_depth,
            "max_pages": max_pages,
            "respect_robots": config.get("crawler.respect_robots", True),
            "delay": config.get("crawler.crawl_delay", 0.5),
            "max_workers": max_workers,  # Add this to crawler config
        }
        self.relevance_threshold = relevance_threshold
        self.model = model.lower()

        # Check for OpenAI API key if using OpenAI
        if self.model == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error(
                    "OpenAI API key not found. Please set OPENAI_API_KEY in environment variables."
                )
                raise ValueError(
                    "OpenAI API key not found. Please set OPENAI_API_KEY in environment variables."
                )

        # Initialize components
        self.vectordb_creator = VectorDBCreator()
        self.rag_handler = RAGHandler(model=self.model)

        # Set up output directory
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(
            f"Initialized RufusExtractor with parallel={use_parallel}, "
            f"max_depth={max_depth}, max_pages={max_pages}, model={model}"
        )

    def process_website(
        self, url: str, instructions: str, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a website through the entire pipeline.

        Args:
            url: Website URL to process
            instructions: User instructions for content extraction
            collection_name: Optional name for vector store collection

        Returns:
            Dict containing processing results and statistics
        """
        try:
            # 1. Set up RelevanceChecker with the selected model
            # Note: Assuming RelevanceChecker has a class method to set model
            if hasattr(RelevanceChecker, "set_model"):
                RelevanceChecker.set_model(self.model)

            print(
                f"=" * 10,
                f"Step 1: Processing {url} with model {self.model}...",
                f"=" * 10,
            )
            # 2. Crawl the website
            logger.info(f"Starting crawl of {url} using {self.model} model")
            crawl_func = crawl_website_parallel if self.use_parallel else crawl_website
            crawl_results = crawl_func(
                url=url, instructions=instructions, **self.crawler_config
            )
            print(f"=" * 10, f"Step 1: crawling completed...", f"=" * 10)

            print(f"=" * 10, f"Step 2: Extracting content...", f"=" * 10)
            # 3. Process extracted content
            relevant_pages = crawl_results["relevant_pages"]
            logger.info(f"Found {len(relevant_pages)} relevant pages")

            # Generate collection name if not provided
            if not collection_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                domain = (
                    url.replace("https://", "").replace("http://", "").split("/")[0]
                )
                collection_name = f"rufus_{domain}_{timestamp}"

            print(f"=" * 10, f"Step 2: Extracting content completed...", f"=" * 10)

            print(f"=" * 10, f"Step 3: Creating vector store...", f"=" * 10)
            # 4. Prepare documents for vector store
            documents = []
            for page_url, content in relevant_pages.items():
                # Convert headings list to string to comply with ChromaDB requirements
                # Handle different heading formats - fix for the error
                headings_str = ""
                headings = content.get("headings", [])

                # Check if headings is a list of dictionaries or a list of strings
                if headings and isinstance(headings, list):
                    if headings and isinstance(headings[0], dict):
                        # If headings are dictionaries with a 'text' key
                        headings_str = " | ".join(h.get("text", "") for h in headings)
                    elif headings and isinstance(headings[0], str):
                        # If headings are directly strings
                        headings_str = " | ".join(headings)

                doc = {
                    "content": content["content"],
                    "url": page_url,
                    "metadata": {
                        "title": content.get("title", ""),
                        "description": content.get("meta_description", ""),
                        "crawl_timestamp": str(content.get("extracted_at", "")),
                        "headings": headings_str,  # Now a string instead of list
                        "source_type": "webpage",
                    },
                }
                documents.append(doc)
            print(f"=" * 10, f"Step 3: Creating vector store completed...", f"=" * 10)

            print(f"=" * 10, f"Step 4: Creating vector store...", f"=" * 10)
            # 5. Create vector store
            logger.info(f"Creating vector store collection: {collection_name}")
            print("Starting vector DB creation...")
            self.vectordb_creator.process_documents(
                documents=documents, collection_name=collection_name
            )
            print(f"=" * 10, f"Step 4: Creating vector store completed...", f"=" * 10)

            print(f"=" * 10, f"Step 5: Setting up RAG...", f"=" * 10)
            # 6. Set up RAG for this collection with the selected model
            logger.info(f"Setting up RAG chain with {self.model} model")

            # Create a RAG handler with the selected model
            self.rag_handler = RAGHandler(model=self.model)

            qa_chain = self.rag_handler.setup_retrieval_qa(
                collection_name=collection_name
            )

            print(f"=" * 10, f"Step 5: Setting up RAG completed...", f"=" * 10)

            print(f"=" * 10, f"Step 6: Saving results...", f"=" * 10)
            # 7. Save processing results
            results = {
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "collection_name": collection_name,
                "model": self.model,
                "stats": {
                    "pages_crawled": len(crawl_results["all_visited_urls"]),
                    "relevant_pages": len(relevant_pages),
                    "documents_processed": len(documents),
                },
                "crawl_results": {
                    "relevant_urls": list(relevant_pages.keys()),
                    "all_visited_urls": crawl_results["all_visited_urls"],
                },
            }

            # Save results to file
            output_file = os.path.join(
                self.output_dir, f"rufus_results_{collection_name}.json"
            )
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Processing complete. Results saved to {output_file}")
            print(f"=" * 10, f"Step 6: Saving results completed...", f"=" * 10)
            print(f"=" * 10, f"Step 7: returns...", f"=" * 10)
            return results

        except Exception as e:
            logger.error("Error processing website", exc_info=True)
            raise

    def query_collection(
        self, collection_name: str, query: str, custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query a processed collection using RAG.

        Args:
            collection_name: Name of the collection to query
            query: User query string
            custom_prompt: Optional custom prompt for the LLM

        Returns:
            Dict containing answer and source information
        """
        try:
            # Set up RAG chain for querying with the selected model
            self.rag_handler = RAGHandler(model=self.model)
            qa_chain = self.rag_handler.setup_retrieval_qa(
                collection_name=collection_name
            )

            # Generate answer
            response = self.rag_handler.generate_answer(
                query=query, qa_chain=qa_chain, custom_prompt=custom_prompt
            )

            logger.info(
                f"Generated answer for query: {query[:50]}... using {self.model} model"
            )
            return response

        except Exception as e:
            logger.error(f"Error querying collection: {collection_name}", exc_info=True)
            raise

    def batch_process_websites(
        self,
        urls: List[str],
        instructions: str,
        collection_prefix: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process multiple websites in batch.

        Args:
            urls: List of website URLs to process
            instructions: User instructions for content extraction
            collection_prefix: Optional prefix for collection names

        Returns:
            List of processing results for each website
        """
        results = []
        for url in urls:
            try:
                # Generate collection name with prefix if provided
                if collection_prefix:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    domain = (
                        url.replace("https://", "").replace("http://", "").split("/")[0]
                    )
                    collection_name = f"{collection_prefix}_{domain}_{timestamp}"
                else:
                    collection_name = None

                # Process website
                result = self.process_website(
                    url=url, instructions=instructions, collection_name=collection_name
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {url}", exc_info=True)
                results.append(
                    {
                        "url": url,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return results


# Example usage
if __name__ == "__main__":
    # Initialize extractor with Ollama model (default)
    rufus = RufusExtractor(
        use_parallel=True, max_depth=1, max_pages=2, model="ollama"  # or "openai"
    )

    # Example website and instructions
    test_url = "https://www.python.org"
    test_instructions = (
        "Extract information about Python features, documentation, and tutorials"
    )

    try:
        # Process website
        results = rufus.process_website(url=test_url, instructions=test_instructions)

        print(f"Processed {results['stats']['pages_crawled']} pages")
        print(f"Found {results['stats']['relevant_pages']} relevant pages")

        # Example query
        collection_name = results["collection_name"]
        query = "What are the main features of Python?"

        answer = rufus.query_collection(collection_name=collection_name, query=query)

        print("\nQuery Results:")
        print(f"Answer: {answer['answer']}")
        print("\nSources:")
        for url in answer["source_urls"]:
            print(f"- {url}")

    except Exception as e:
        print(f"Error in example: {str(e)}")
