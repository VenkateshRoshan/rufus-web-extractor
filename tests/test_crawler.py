"""
Unit tests for the web crawler module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rufus_2.crawler import Crawler


class TestCrawler(unittest.TestCase):
    """Tests for the Crawler class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a crawler with low limits for testing
        self.crawler = Crawler(
            delay=0.01,  # Low delay for tests
            max_depth=2,
            max_pages=5,
            same_domain_only=True,
            respect_robots_txt=False,  # Disable for tests
        )

        # Mock requests.get
        self.requests_patcher = patch("rufus.crawler.requests.get")
        self.mock_requests_get = self.requests_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        self.requests_patcher.stop()

    def test_get_domain(self):
        """Test the _get_domain method."""
        # Test with different URLs
        self.assertEqual(
            self.crawler._get_domain("https://www.python.org"), "www.python.org"
        )
        self.assertEqual(
            self.crawler._get_domain("https://www.python.org/path"), "www.python.org"
        )
        self.assertEqual(
            self.crawler._get_domain("https://sub.example.com/path?query=test"),
            "sub.example.com",
        )

    def test_extract_links(self):
        """Test the _extract_links method."""
        # Sample HTML content with links
        html_content = """
        <html>
            <body>
                <a href="https://www.python.org/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="#section">Section</a>
                <a href="https://other-domain.com/page">External Page</a>
            </body>
        </html>
        """

        # Test link extraction
        links = self.crawler._extract_links("https://www.python.org", html_content)

        # Should find 2 links (one absolute, one relative, excluding fragment and external domain)
        self.assertEqual(len(links), 2)
        self.assertIn("https://www.python.org/page1", links)
        self.assertIn("https://www.python.org/page2", links)
        self.assertNotIn(
            "https://www.python.org#section", links
        )  # Fragment should be excluded
        self.assertNotIn(
            "https://other-domain.com/page", links
        )  # External domain should be excluded

    def test_extract_text(self):
        """Test the _extract_text method."""
        # Sample HTML content
        html_content = """
        <html>
            <head>
                <title>Test Page</title>
                <style>body { color: black; }</style>
                <script>console.log('Hello');</script>
            </head>
            <body>
                <h1>Heading</h1>
                <p>This is a paragraph with <b>bold</b> text.</p>
            </body>
        </html>
        """

        # Extract text
        text = self.crawler._extract_text(html_content)

        # Check that we got the text content, without script or style
        self.assertIn("Heading", text)
        self.assertIn("This is a paragraph with bold text", text.replace("\n", " "))
        self.assertNotIn("console.log", text)
        self.assertNotIn("color: black", text)

    def test_crawl(self):
        """Test the crawl method."""
        # Set up mock response for initial URL
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <h1>Test Page</h1>
                <p>This is a test page.</p>
                <a href="https://www.python.org/page1">Page 1</a>
                <a href="https://www.python.org/page2">Page 2</a>
            </body>
        </html>
        """

        # Set up mock response for sub-pages
        mock_subpage_response = MagicMock()
        mock_subpage_response.status_code = 200
        mock_subpage_response.text = """
        <html>
            <body>
                <h1>Sub Page</h1>
                <p>This is a sub-page.</p>
            </body>
        </html>
        """

        # Configure mock to return different responses
        self.mock_requests_get.side_effect = lambda url, **kwargs: (
            mock_response if url == "https://www.python.org" else mock_subpage_response
        )

        # Crawl starting from example.com
        results = self.crawler.crawl("https://www.python.org")

        # Should have crawled the initial page and 2 sub-pages (up to max_pages=5)
        self.assertEqual(len(results), 3)

        # Check that we got the URLs we expected
        crawled_urls = [url for url, _, _ in results]
        self.assertIn("https://www.python.org", crawled_urls)
        self.assertIn("https://www.python.org/page1", crawled_urls)
        self.assertIn("https://www.python.org/page2", crawled_urls)

        # Verify requests.get was called with the expected URLs
        expected_calls = [
            unittest.mock.call(
                "https://www.python.org", headers=unittest.mock.ANY, timeout=10
            ),
            unittest.mock.call(
                "https://www.python.org/page1", headers=unittest.mock.ANY, timeout=10
            ),
            unittest.mock.call(
                "https://www.python.org/page2", headers=unittest.mock.ANY, timeout=10
            ),
        ]
        self.mock_requests_get.assert_has_calls(expected_calls, any_order=True)


if __name__ == "__main__":
    unittest.main()
