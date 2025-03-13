"""
Gradio Web Interface for Rufus Web Extractor (API version)

This module provides a Gradio-based web UI for interacting with the Rufus web extraction tool.
It uses the Rufus API endpoints instead of importing the RufusExtractor directly.
"""

import os
import json
import time
import gradio as gr
import requests
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Import logger and configuration
from rufus.helpers.logger import logger
from rufus.config import config

# Global variables
API_BASE_URL = config.get("api.url", "http://localhost:8234")
API_HOST = config.get("api.host", "0.0.0.0")
API_PORT = config.get("api.port", 8234)
GRADIO_HOST = config.get("gradio.host", "0.0.0.0")
GRADIO_PORT = config.get("gradio.port", 7860)
GRADIO_SHARE = config.get("gradio.share", False)

current_collection = None
current_url = None
processing_status = "idle"
results_data = None
current_job_id = None


def check_url(url: str) -> Tuple[bool, str]:
    """
    Check if a URL is valid and accessible.

    Args:
        url: URL to check

    Returns:
        Tuple of (is_valid, message)
    """
    # Ensure URL has proper scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        # Send a HEAD request to check if the URL is accessible
        response = requests.head(url, timeout=10)
        if response.status_code < 400:
            return True, url
        else:
            return False, f"URL returned status code {response.status_code}"
    except Exception as e:
        return False, f"Error accessing URL: {str(e)}"


def process_website(url: str, max_depth: int, max_pages: int) -> Dict[str, Any]:
    """
    Process a website using the Rufus API.

    Args:
        url: Website URL to process
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl

    Returns:
        Processing results
    """
    global current_collection, current_url, processing_status, results_data, current_job_id

    # Ensure URL has proper scheme
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Set current URL
    current_url = url

    # Generate instructions based on URL
    domain = url.replace("https://", "").replace("http://", "").split("/")[0]

    # Use the custom instructions template from config if available
    instructions = config.get(
        "instructions_template",
        f"Extract comprehensive information from {domain} website",
    )

    try:
        # Create the scrape request
        headers = {"Content-Type": "application/json"}
        payload = {
            "url": url,
            "instructions": instructions,
            "max_depth": max_depth,
            "max_pages": max_pages,
        }

        # Call the API to initiate the scraping
        logger.info(f"Calling API to scrape: {url}")
        processing_status = "processing"

        response = requests.post(
            f"{API_BASE_URL}/scrape", headers=headers, json=payload
        )

        if response.status_code != 200:
            logger.error(f"API error: {response.status_code} - {response.text}")
            raise Exception(f"API returned status code {response.status_code}")

        # Get job details
        job_data = response.json()
        job_id = job_data.get("job_id")
        current_job_id = job_id

        logger.info(f"Scrape job created: {job_id}")

        # Poll for job completion
        while job_data.get("status") in ["pending", "processing"]:
            time.sleep(2)  # Wait for 2 seconds before checking again
            job_response = requests.get(
                f"{API_BASE_URL}/jobs/{job_id}", headers=headers
            )

            if job_response.status_code != 200:
                raise Exception("Failed to retrieve job status")

            job_data = job_response.json()

            # Break if job is completed or failed
            if job_data.get("status") in ["completed", "failed"]:
                break

        # Update status based on job completion
        if job_data.get("status") == "completed":
            processing_status = "completed"
            current_collection = (
                job_data.get("collection_name")
                or job_data.get("result_path", "").split("/")[-1].split("_")[1]
            )

            # Try to get detailed results if available
            if job_data.get("result_path"):
                try:
                    with open(job_data.get("result_path"), "r") as f:
                        results_data = json.load(f)
                except Exception as e:
                    logger.error(f"Error loading results file: {str(e)}")
                    # Create a basic results structure
                    results_data = {
                        "url": url,
                        "collection_name": current_collection,
                        "timestamp": datetime.now().isoformat(),
                        "job_id": job_id,
                        "stats": {
                            "pages_crawled": 0,
                            "relevant_pages": 0,
                            "documents_processed": 0,
                        },
                    }
            else:
                # Create a basic results structure
                results_data = {
                    "url": url,
                    "collection_name": current_collection,
                    "timestamp": datetime.now().isoformat(),
                    "job_id": job_id,
                    "stats": {
                        "pages_crawled": 0,
                        "relevant_pages": 0,
                        "documents_processed": 0,
                    },
                }
        else:
            # Job failed
            error_message = job_data.get("error", "Unknown error")
            logger.error(f"Scrape job failed: {error_message}")
            raise Exception(f"Scraping failed: {error_message}")

        return results_data

    except Exception as e:
        logger.error(f"Error processing website: {str(e)}", exc_info=True)
        processing_status = "failed"
        raise


def query_collection(query: str) -> Dict[str, Any]:
    """
    Query the current collection using the Rufus API.

    Args:
        query: User query

    Returns:
        Dict containing answer and source information
    """
    global current_collection

    if not current_collection:
        return {
            "answer": "Please process a website first before querying.",
            "source_urls": [],
        }

    try:
        # Call the API to query the collection
        headers = {"Content-Type": "application/json"}
        payload = {"collection_name": current_collection, "query": query}

        logger.info(f"Querying collection '{current_collection}' with query: {query}")
        response = requests.post(f"{API_BASE_URL}/query", headers=headers, json=payload)

        if response.status_code != 200:
            logger.error(
                f"API error during query: {response.status_code} - {response.text}"
            )
            return {
                "answer": f"Error querying API: HTTP {response.status_code}",
                "source_urls": [],
            }

        # Parse the response
        api_response = response.json()

        # Make sure we have expected return values
        if not isinstance(api_response, dict):
            logger.error(f"Response is not a dictionary: {api_response}")
            return {
                "answer": "The system returned an unexpected response format.",
                "source_urls": [],
            }

        # Ensure we have an answer field
        if "answer" not in api_response:
            logger.warning("Response missing 'answer' field")
            api_response[
                "answer"
            ] = "The system generated a response without an answer field."

        # Ensure we have source_urls field
        if "source_urls" not in api_response:
            logger.warning("Response missing 'source_urls' field")
            api_response["source_urls"] = []

        logger.info("Query successful, returning response")
        return api_response

    except Exception as e:
        # Handle any errors
        error_message = f"Error querying collection: {str(e)}"
        logger.error(error_message)
        return {
            "answer": f"An error occurred while processing your query: {str(e)}",
            "source_urls": [],
        }


def format_source_urls(source_urls: List[str]) -> str:
    """Format source URLs as markdown links."""
    if not source_urls:
        return ""

    formatted = "\n\n**Sources:**\n"
    for i, url in enumerate(source_urls, 1):
        formatted += f"{i}. [{url}]({url})\n"

    return formatted


def download_results() -> str:
    """
    Create a downloadable results file.

    Returns:
        Path to the results file
    """
    global results_data, current_url

    if not results_data or not current_url:
        return None

    # Create output directory if it doesn't exist
    os.makedirs("downloads", exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    domain = current_url.replace("https://", "").replace("http://", "").split("/")[0]
    filename = f"rufus_results_{domain}_{timestamp}.json"
    filepath = os.path.join("downloads", filename)

    # Save data
    with open(filepath, "w") as f:
        json.dump(results_data, f, indent=2)

    return filepath


def validate_url_and_show_chat(url):
    """
    Validate the URL and make the chat interface visible if it's valid.
    Does NOT start any processing yet.

    Args:
        url: URL to validate

    Returns:
        Updated UI components
    """
    # Check if URL is empty
    if not url.strip():
        return (
            "⚪ Please enter a URL first.",  # Status message
            gr.update(visible=False),  # Processing indicators
            gr.update(visible=False),  # Chat container
            gr.update(visible=False),  # Stats container
            None,  # Download output
        )

    # Validate URL
    is_valid, url_or_message = check_url(url)

    if not is_valid:
        return (
            f"❌ Error: {url_or_message}",  # Status message
            gr.update(visible=False),  # Processing indicators
            gr.update(visible=False),  # Chat container
            gr.update(visible=False),  # Stats container
            None,  # Download output
        )

    # URL is valid, set the global variable
    global current_url
    current_url = url_or_message

    # Show chat interface but don't start processing
    return (
        "🟢 URL is valid. Please enter your query in the chat box below.",  # Status message
        gr.update(visible=False),  # Processing indicators
        gr.update(visible=True),  # Chat container (now visible)
        gr.update(visible=False),  # Stats container
        None,  # Download output
    )


def user_query(
    user_message: str, chat_history, url: str, max_depth: int, max_pages: int
):
    """
    Handle user query, process website if needed, and update chat history.
    Uses the proper message format with 'role' and 'content' fields.
    Uses the API for processing.

    Args:
        user_message: User's question
        chat_history: Current chat history
        url: URL to process
        max_depth: Maximum crawl depth
        max_pages: Maximum pages to crawl

    Returns:
        Updated chat history
    """
    global processing_status, current_collection

    logger.info(f"Processing query: '{user_message}'")

    if not user_message.strip():
        return chat_history

    # Create a new list so we don't modify the original
    new_history = list(chat_history) if chat_history else []

    # Add user message to history
    new_history.append({"role": "user", "content": user_message})
    new_history.append({"role": "assistant", "content": "🔄 Processing your request..."})
    yield new_history

    try:
        # Ensure URL has proper scheme
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        # If the website hasn't been processed yet, process it now
        if processing_status != "completed" or not current_collection:
            # Show crawling status
            new_history[-1] = {
                "role": "assistant",
                "content": "🔍 First crawling the website. This may take a few minutes...",
            }
            yield new_history

            # Process the website using the API
            try:
                logger.info(f"Starting to process website via API: {url}")
                process_website(url, max_depth, max_pages)
                new_history[-1] = {
                    "role": "assistant",
                    "content": "✅ Website processed successfully. Now generating answer...",
                }
                yield new_history
                logger.info("Website processing completed")
            except Exception as e:
                error_message = f"❌ Error processing website: {str(e)}"
                logger.error(f"Website processing error: {str(e)}")
                new_history[-1] = {"role": "assistant", "content": error_message}
                yield new_history
                return

        # Show processing status for query
        for status in [
            "🔍 Searching relevant content...",
            "🤔 Thinking...",
            "✍️ Generating answer...",
        ]:
            new_history[-1] = {"role": "assistant", "content": status}
            yield new_history
            time.sleep(0.5)

        # Get answer using the API
        try:
            logger.info("Getting answer from API")
            response = query_collection(user_message)

            answer = response.get("answer", "No answer was generated.")
            logger.info(f"Got answer of length {len(answer)}")

            sources = format_source_urls(response.get("source_urls", []))
            final_answer = answer + sources

            # Update the last message with the final answer
            new_history[-1] = {"role": "assistant", "content": final_answer}
            logger.info(f"Final answer length: {len(final_answer)}")
            # Use yield for the final answer to ensure it updates the UI
            yield new_history

        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            error_message = f"⚠️ Error generating response: {str(e)}"
            new_history[-1] = {"role": "assistant", "content": error_message}
            yield new_history

    except Exception as e:
        logger.error(f"General error in user_query: {str(e)}")
        error_message = f"❌ Error: {str(e)}"
        new_history[-1] = {"role": "assistant", "content": error_message}
        yield new_history


def update_stats():
    """Get statistics from the API about the current collection/job."""
    global results_data, current_job_id

    if not results_data:
        return "Process a website to see statistics."

    try:
        # If we have local results data, use it
        stats = results_data.get("stats", {})
        content = f"""
        ## Website Processing Statistics
        
        - **URL**: {results_data.get("url", "Unknown")}
        - **Collection Name**: {results_data.get("collection_name", "Unknown")}
        - **Job ID**: {current_job_id or "Unknown"}
        - **Timestamp**: {results_data.get("timestamp", "Unknown")}
        
        ### Crawl Statistics
        - Pages Crawled: {stats.get("pages_crawled", "Unknown")}
        - Relevant Pages Found: {stats.get("relevant_pages", "Unknown")}
        - Documents Processed: {stats.get("documents_processed", "Unknown")}
        """

        # Add relevant URLs if available
        if (
            "crawl_results" in results_data
            and "relevant_urls" in results_data["crawl_results"]
        ):
            content += "\n### Relevant URLs\n"
            for i, url in enumerate(results_data["crawl_results"]["relevant_urls"], 1):
                content += f"{i}. [{url}]({url})\n"

        return content

    except Exception as e:
        logger.error(f"Error updating stats: {str(e)}")
        return f"Error retrieving statistics: {str(e)}"


def update_status_after_query():
    return "🟢 Please enter another query"


def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True
        return False
    except:
        return False


def create_ui():
    """Create the Gradio UI for Rufus Web Extractor."""

    # Check API health
    api_running = check_api_health()
    if not api_running:
        logger.warning(
            f"API not accessible at {API_BASE_URL}. Some features may not work."
        )

    # Custom CSS for button styling and layout adjustments
    custom_css = """
    #url-input-container {
        position: relative;
    }
    .url-input-with-button {
        padding-right: 40px !important;
    }
    .round-button {
        position: absolute !important;
        right: 5px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        max-width: 32px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        z-index: 10 !important;
        aspect-ratio: 1 / 1 !important;
        line-height: 1 !important;
    }
    .status-area {
        background-color: transparent !important;
        padding: 4px !important;
        text-align: center !important;
        margin-top: 0 !important;
    }
    /* Reduce spacing between rows in the grid */
    .gap {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Tighten up the spacing between elements */
    .container {
        gap: 0 !important;
    }
    .row > .gr-form {
        margin-bottom: 4px !important;
    }
    """

    with gr.Blocks(
        title="Rufus Web Extractor", theme=gr.themes.Soft(), css=custom_css
    ) as app:
        gr.Markdown("# 🌐 Rufus Web Extractor")
        gr.Markdown("Extract and query information from websites using AI.")

        # Add API status indicator
        api_status = (
            "🟢 API Connected"
            if api_running
            else "🔴 API Not Connected (Using fallback mode)"
        )
        gr.Markdown(f"**API Status:** {api_status}", elem_id="api-status")

        # Create the 2x2 grid layout
        with gr.Row():
            # First column (URL input + Status)
            with gr.Column(scale=2):
                # URL input with button (row 1, col 1)
                with gr.Row(elem_id="url-input-container"):
                    url_input = gr.Textbox(
                        show_label=False,
                        placeholder="Enter a website URL (e.g., https://www.python.org)",
                        lines=1,
                        elem_classes="url-input-with-button",
                    )
                    # Round button with arrow inside the textbox
                    process_button = gr.Button(
                        "➤", elem_classes="round-button", variant="primary", size="sm"
                    )

                # Status area (row 2, col 1)
                with gr.Row():
                    status_area = gr.Markdown(
                        "⚪ Enter a URL and click the arrow button to begin.",
                        elem_classes="status-area",
                    )

            # Second column (Parameters)
            with gr.Column(scale=1):
                # Parameters sliders (row 1+2, col 2)
                max_depth = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=config.get("crawler.max_depth", 2),
                    step=1,
                    label="Max Depth",
                    show_label=True,
                    interactive=True,
                )

                max_pages = gr.Slider(
                    minimum=2,
                    maximum=100,
                    value=config.get("crawler.max_pages", 5),
                    step=1,
                    label="Max Pages",
                    show_label=True,
                    interactive=True,
                )

        # Processing status indicators (initially hidden)
        with gr.Column(visible=False) as processing_indicators:
            with gr.Row():
                crawling_status = gr.Markdown("⚪ Crawling Website")
                extracting_status = gr.Markdown("⚪ Extracting Content")
                vectordb_status = gr.Markdown("⚪ Creating Vector Store")
                ready_status = gr.Markdown("⚪ Ready for Queries")

        # Chat interface (initially hidden)
        with gr.Column(visible=False) as chat_container:
            gr.Markdown("## Ask Questions About the Website")

            chat_interface = gr.Chatbot(
                height=500,
                show_label=False,
                avatar_images=("🧑", "🤖"),
                type="messages",  # Using messages instead of tuples format
            )

            with gr.Row():
                with gr.Column(scale=4):
                    msg = gr.Textbox(
                        label="Ask a question",
                        placeholder="What would you like to know about this website?",
                        lines=1,
                        show_label=False,
                    )

                with gr.Column(scale=2, min_width=100):
                    submit_btn = gr.Button("Send", variant="primary")

            with gr.Row():
                clear_btn = gr.Button("Clear Chat")
                download_btn = gr.Button("Download Data")

        # Statistics area (initially hidden)
        with gr.Column(visible=False) as stats_container:
            gr.Markdown("## Website Statistics")
            stats_display = gr.Markdown("Processing website...")

        # Download output
        download_output = gr.File(label="Download Results", visible=False)

        # Event handlers

        # Just validate URL and show chat interface when button is clicked
        process_button.click(
            validate_url_and_show_chat,
            inputs=url_input,
            outputs=[
                status_area,
                processing_indicators,
                chat_container,
                stats_container,
                download_output,
            ],
        )

        # Process query and website when send button is clicked
        submit_btn.click(
            user_query,
            inputs=[msg, chat_interface, url_input, max_depth, max_pages],
            outputs=chat_interface,
            queue=True,
        ).then(lambda: "", outputs=msg).then(
            update_status_after_query, outputs=status_area
        )

        # Also process on Enter key
        msg.submit(
            user_query,
            inputs=[msg, chat_interface, url_input, max_depth, max_pages],
            outputs=chat_interface,
            queue=True,
        ).then(lambda: "", outputs=msg).then(
            update_status_after_query, outputs=status_area
        )

        clear_btn.click(lambda: [], outputs=chat_interface)

        download_btn.click(download_results, outputs=download_output)

        # Also handle updating statistics
        download_btn.click(update_stats, outputs=stats_display).then(
            lambda: gr.update(visible=True), outputs=stats_container
        )

    return app


# Update the launch section
def main():
    """Main function to launch the Gradio app."""
    app = create_ui()
    try:
        # Use configuration from config
        app.launch(
            server_name=GRADIO_HOST,
            server_port=GRADIO_PORT,
            share=GRADIO_SHARE,
        )
    except Exception as e:
        logger.error(f"Error launching Gradio app: {str(e)}")
        print(f"Failed to launch Gradio app: {str(e)}")


if __name__ == "__main__":
    main()
