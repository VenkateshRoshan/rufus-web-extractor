"""
Search Module - Handles content extraction from web pages.

This module provides functionality to extract useful content from web pages.
It focuses on cleaning and structuring the content for further processing.
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, List, Union
import re
import time
from urllib.parse import urlparse

# Import logger
from rufus.helpers.logger import logger


def search_content(url: str) -> Dict[str, Any]:
    """
    Extract and structure content from a web page.

    Args:
        url (str): URL of the page to extract content from.

    Returns:
        Dict[str, Any]: Structured content from the page.
    """
    logger.info(f"Extracting content from {url}")

    try:
        # Fetch the page
        headers = {
            "User-Agent": "Rufus-Web-Extractor/1.0 (+https://github.com/your-username/rufus)"
        }
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return {}

        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract structured content
        structured_content = extract_structured_content(soup, url)

        logger.info(f"Successfully extracted structured content from {url}")
        return structured_content

    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}", exc_info=True)
        return {}


def extract_structured_content(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    """
    Extract structured content from BeautifulSoup object.

    Args:
        soup (BeautifulSoup): BeautifulSoup object of the page.
        url (str): URL of the page.

    Returns:
        Dict[str, Any]: Structured content.
    """
    # Initialize the content dictionary
    content = {
        "url": url,
        "timestamp": time.time(),
        "domain": urlparse(url).netloc,
    }

    # Extract title
    content["title"] = extract_title(soup)

    # Extract meta description
    content["meta_description"] = extract_meta_description(soup)

    # Extract author information
    content["author"] = extract_author(soup)

    # Extract publish date
    content["publish_date"] = extract_publish_date(soup)

    # Extract main content
    content["main_content"] = extract_main_content(soup)

    # Extract structured data (JSON-LD, etc.)
    content["structured_data"] = extract_structured_data(soup)

    # Extract headings hierarchy
    content["headings"] = extract_headings(soup)

    # Extract links with context
    content["links"] = extract_links_with_context(soup, url)

    # Extract images with alt text and captions
    content["images"] = extract_images(soup)

    # Extract tables
    content["tables"] = extract_tables(soup)

    # Extract lists
    content["lists"] = extract_lists(soup)

    # Extract code blocks
    content["code_blocks"] = extract_code_blocks(soup)

    # Extract main text content (cleaned)
    content["content"] = clean_content(content["main_content"])

    return content


def extract_title(soup: BeautifulSoup) -> str:
    """Extract the title of the page."""
    title = ""

    # Try to get the title from the title tag
    if soup.title:
        title = soup.title.string

    # If no title found, try h1
    if not title and soup.h1:
        title = soup.h1.get_text(strip=True)

    # Clean the title
    if title:
        title = title.strip()

    return title or "No Title"


def extract_meta_description(soup: BeautifulSoup) -> str:
    """Extract the meta description of the page."""
    meta_desc = ""

    # Try to get meta description
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag:
        meta_desc = meta_tag.get("content", "")

    # If no meta description, try Open Graph description
    if not meta_desc:
        og_desc = soup.find("meta", property="og:description")
        if og_desc:
            meta_desc = og_desc.get("content", "")

    return meta_desc.strip() if meta_desc else ""


def extract_author(soup: BeautifulSoup) -> str:
    """Extract the author of the content."""
    author = ""

    # Try different common patterns for author information
    author_patterns = [
        ("meta", {"name": "author"}),
        ("meta", {"property": "article:author"}),
        ("a", {"rel": "author"}),
        ("span", {"class": "author"}),
        ("div", {"class": "author"}),
        ("p", {"class": "byline"}),
    ]

    for tag, attrs in author_patterns:
        element = soup.find(tag, attrs)
        if element:
            if tag == "meta":
                author = element.get("content", "")
            else:
                author = element.get_text(strip=True)

            if author:
                break

    return author.strip() if author else ""


def extract_publish_date(soup: BeautifulSoup) -> str:
    """Extract the publish date of the content."""
    publish_date = ""

    # Try different common patterns for publish date
    date_patterns = [
        ("meta", {"name": "article:published_time"}),
        ("meta", {"property": "article:published_time"}),
        ("meta", {"name": "publication-date"}),
        ("meta", {"name": "publish-date"}),
        ("time", {"class": "published"}),
        ("span", {"class": "published"}),
        ("p", {"class": "date"}),
    ]

    for tag, attrs in date_patterns:
        element = soup.find(tag, attrs)
        if element:
            if tag == "meta":
                publish_date = element.get("content", "")
            else:
                publish_date = element.get_text(strip=True)

            if publish_date:
                break

    return publish_date.strip() if publish_date else ""


def extract_main_content(soup: BeautifulSoup) -> str:
    """
    Extract the main content of the page.
    This uses heuristics to find the most likely content container.
    """
    # First, remove noise elements
    for noise in soup.select(
        "script, style, header, footer, nav, .nav, .navigation, .header, .footer, .sidebar, .menu, .comments, .ad, .advertisement, aside, form"
    ):
        noise.extract()

    # Try to find main content by common container patterns
    content_containers = [
        "article",
        "main",
        'div[role="main"]',
        ".content",
        "#content",
        ".post",
        ".entry",
        ".post-content",
        ".article-content",
        ".entry-content",
        ".story",
        ".post-body",
    ]

    main_content = ""

    # Try to find content by selector
    for container in content_containers:
        elements = soup.select(container)
        if elements:
            # Take the largest element by text length as it's likely the main content
            main_element = max(elements, key=lambda el: len(el.get_text(strip=True)))
            main_content = main_element.get_text(separator=" ", strip=True)
            break

    # If no content found, use the body but try to be smart about it
    if not main_content and soup.body:
        # Get all paragraphs
        paragraphs = soup.find_all("p")

        # If we have paragraphs, join them (common in articles)
        if paragraphs:
            main_content = " ".join(
                p.get_text(strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True)) > 20
            )
        else:
            # If no paragraphs, use the body text but try to clean it
            main_content = soup.body.get_text(separator=" ", strip=True)

    return main_content


def extract_structured_data(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract structured data (JSON-LD, microdata, etc.)."""
    structured_data = []

    # Try to find JSON-LD
    json_ld_scripts = soup.find_all("script", type="application/ld+json")

    import json

    for script in json_ld_scripts:
        try:
            data = json.loads(script.string)
            structured_data.append(data)
        except (json.JSONDecodeError, TypeError):
            continue

    return structured_data


def extract_headings(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Extract headings hierarchy from the page."""
    headings = []

    for level in range(1, 7):  # h1 to h6
        for heading in soup.find_all(f"h{level}"):
            headings.append({"level": level, "text": heading.get_text(strip=True)})

    return headings


def extract_links_with_context(
    soup: BeautifulSoup, base_url: str
) -> List[Dict[str, str]]:
    """Extract links with their surrounding context."""
    links = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(strip=True)

        # Skip empty or javascript links
        if not href or href.startswith("javascript:") or href == "#":
            continue

        # Get some context around the link
        parent = a.parent
        context = parent.get_text(strip=True) if parent else ""

        # If context is too long, trim it
        if len(context) > 200:
            # Try to center it around the link text
            link_pos = context.find(text)
            if link_pos >= 0:
                start = max(0, link_pos - 100)
                end = min(len(context), link_pos + len(text) + 100)
                context = context[start:end]
                if start > 0:
                    context = "..." + context
                if end < len(parent.get_text(strip=True)):
                    context = context + "..."

        links.append({"url": href, "text": text, "context": context})

    return links


def extract_images(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Extract images with alt text and captions."""
    images = []

    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src:
            continue

        alt_text = img.get("alt", "")

        # Try to find a caption
        caption = ""

        # Check if image is inside a figure with figcaption
        figure_parent = img.find_parent("figure")
        if figure_parent:
            figcaption = figure_parent.find("figcaption")
            if figcaption:
                caption = figcaption.get_text(strip=True)

        # If no figcaption found, try nearby elements that might be captions
        if not caption:
            # Look for adjacent or parent elements with caption-like classes
            caption_elements = img.find_next_siblings(
                class_=re.compile("caption|wp-caption-text")
            )
            if caption_elements:
                caption = caption_elements[0].get_text(strip=True)

        images.append({"src": src, "alt": alt_text, "caption": caption})

    return images


def extract_tables(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract tables from the page."""
    tables = []

    for table in soup.find_all("table"):
        table_data = {"headers": [], "rows": []}

        # Extract table headers
        headers = table.find_all("th")
        if headers:
            table_data["headers"] = [th.get_text(strip=True) for th in headers]

        # Extract table rows
        for tr in table.find_all("tr"):
            # Skip header rows
            if tr.find("th"):
                continue

            row = [td.get_text(strip=True) for td in tr.find_all("td")]
            if row:
                table_data["rows"].append(row)

        # Only add tables with actual data
        if table_data["rows"]:
            # Try to find a caption or title for the table
            caption = table.find("caption")
            if caption:
                table_data["caption"] = caption.get_text(strip=True)

            tables.append(table_data)

    return tables


def extract_lists(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Extract lists from the page."""
    lists = []

    for list_type in ["ul", "ol"]:
        for list_element in soup.find_all(list_type):
            list_data = {
                "type": "unordered" if list_type == "ul" else "ordered",
                "items": [],
            }

            for li in list_element.find_all("li", recursive=False):
                list_data["items"].append(li.get_text(strip=True))

            # Only add non-empty lists
            if list_data["items"]:
                lists.append(list_data)

    return lists


def extract_code_blocks(soup: BeautifulSoup) -> List[Dict[str, str]]:
    """Extract code blocks from the page."""
    code_blocks = []

    # Find code blocks in pre>code structure
    for pre in soup.find_all("pre"):
        code = pre.find("code")
        if code:
            language = ""
            # Try to determine the language
            class_attr = code.get("class", [])
            if class_attr:
                for cls in class_attr:
                    if cls.startswith("language-") or cls.startswith("lang-"):
                        language = cls.split("-", 1)[1]
                        break

            code_blocks.append({"language": language, "code": code.get_text()})

    # Find code blocks marked with class
    for code in soup.find_all("code", class_=re.compile("language-|lang-|prettyprint")):
        language = ""
        class_attr = code.get("class", [])
        if class_attr:
            for cls in class_attr:
                if cls.startswith("language-") or cls.startswith("lang-"):
                    language = cls.split("-", 1)[1]
                    break

        code_blocks.append({"language": language, "code": code.get_text()})

    return code_blocks


def clean_content(content: str) -> str:
    """Clean the extracted content."""
    if not content:
        return ""

    # Remove excessive whitespace
    content = re.sub(r"\s+", " ", content)

    # Remove common noise phrases
    noise_phrases = [
        "Cookie Notice",
        "Accept Cookies",
        "Privacy Policy",
        "Terms of Service",
        "Login",
        "Sign Up",
        "Subscribe",
        "Newsletter",
        "Share this",
        "Share on",
        "Follow us",
        "Related Articles",
        "Read more",
        "Load more",
    ]

    for phrase in noise_phrases:
        content = re.sub(re.escape(phrase), "", content, flags=re.IGNORECASE)

    # Trim and return
    return content.strip()


# Function to search for specific information in content
def search_in_content(content: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
    """
    Search for specific information in content based on a query.

    Args:
        content (Dict[str, Any]): Content to search in.
        query (str): Search query.

    Returns:
        List[Dict[str, Any]]: Search results with relevance scores.
    """
    logger.info(f"Searching for '{query}' in content")

    results = []

    # Simple keyword matching for now
    # In a real implementation, this would use more sophisticated techniques
    # like vector search or retrieval with an LLM

    query_terms = query.lower().split()

    # Search in main content
    if "content" in content:
        text_content = content["content"].lower()

        # Calculate a simple relevance score based on term frequency
        score = sum(text_content.count(term) for term in query_terms)

        if score > 0:
            snippet = extract_snippet(content["content"], query)
            results.append({"type": "main_content", "score": score, "snippet": snippet})

    # Search in headings
    if "headings" in content and content["headings"]:
        headings_text = " ".join(h["text"].lower() for h in content["headings"])

        score = sum(headings_text.count(term) for term in query_terms)

        if score > 0:
            matching_headings = [
                h
                for h in content["headings"]
                if any(term in h["text"].lower() for term in query_terms)
            ]

            results.append(
                {
                    "type": "headings",
                    "score": score * 2,  # Headings are more important
                    "matches": matching_headings,
                }
            )

    # Return sorted results by score
    return sorted(results, key=lambda x: x["score"], reverse=True)


def extract_snippet(text: str, query: str, context_size: int = 100) -> str:
    """
    Extract a snippet of text around the query match.

    Args:
        text (str): Text to extract snippet from.
        query (str): Query to find in text.
        context_size (int): Number of characters to include around the match.

    Returns:
        str: Snippet with context around the match.
    """
    if not text or not query:
        return ""

    # Find the position of the query in the text (case-insensitive)
    query_terms = query.lower().split()
    text_lower = text.lower()

    # Find the best match for the first term
    best_pos = -1
    highest_score = 0

    for term in query_terms:
        pos = text_lower.find(term)
        if pos >= 0:
            # Count how many other terms are near this one
            score = 1
            for other_term in [t for t in query_terms if t != term]:
                if text_lower[pos : pos + 200].find(other_term) >= 0:
                    score += 1

            if score > highest_score:
                highest_score = score
                best_pos = pos

    if best_pos >= 0:
        # Extract context around the match
        start = max(0, best_pos - context_size)
        end = min(len(text), best_pos + context_size)

        snippet = text[start:end]

        # Add ellipsis if we truncated
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    # If no match found, return the beginning of the text
    return text[:200] + ("..." if len(text) > 200 else "")


# Example usage
if __name__ == "__main__":
    test_url = "https://www.python.org"
    content = search_content(test_url)

    print(f"Title: {content.get('title', 'No title')}")
    print(f"Description: {content.get('meta_description', 'No description')}")
    print(f"Found {len(content.get('headings', []))} headings")
    print(f"Found {len(content.get('links', []))} links")
    print(f"Content length: {len(content.get('content', ''))}")

    # Search for a specific query
    results = search_in_content(content, "download python")

    print("\nSearch results:")
    for result in results:
        print(f"Type: {result['type']}")
        print(f"Score: {result['score']}")

        if "snippet" in result:
            print(f"Snippet: {result['snippet']}\n")
        elif "matches" in result:
            print("Matching headings:")
            for heading in result["matches"]:
                print(f"  - {heading['text']}")
            print()
