"""
Unit tests for the RufusClient class.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rufus_2.client import RufusClient


class TestRufusClient(unittest.TestCase):
    """Tests for the RufusClient class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create patches for the components
        self.crawler_patcher = patch("rufus.client.Crawler")
        self.ai_processor_patcher = patch("rufus.client.AIProcessor")
        self.doc_synth_patcher = patch("rufus.client.DocumentSynthesizer")

        # Start the patches
        self.mock_crawler = self.crawler_patcher.start()
        self.mock_ai_processor = self.ai_processor_patcher.start()
        self.mock_doc_synth = self.doc_synth_patcher.start()

        # Configure mocks
        self.mock_crawler_instance = MagicMock()
        self.mock_ai_processor_instance = MagicMock()
        self.mock_doc_synth_instance = MagicMock()

        self.mock_crawler.return_value = self.mock_crawler_instance
        self.mock_ai_processor.return_value = self.mock_ai_processor_instance
        self.mock_doc_synth.return_value = self.mock_doc_synth_instance

        # Create client
        self.client = RufusClient(api_key="dummy_key", output_dir="./test_output")

    def tearDown(self):
        """Tear down test fixtures."""
        self.crawler_patcher.stop()
        self.ai_processor_patcher.stop()
        self.doc_synth_patcher.stop()

    def test_scrape(self):
        """Test the scrape method."""
        # Configure mocks for the test
        self.mock_crawler_instance.crawl.return_value = [
            ("https://www.python.org", "<html>Test</html>", "Test content")
        ]

        self.mock_ai_processor_instance.analyze_relevance.return_value = (
            0.8,
            "Relevant",
        )
        self.mock_ai_processor_instance.extract_key_info.return_value = {"key": "value"}

        self.mock_doc_synth_instance.process_extracted_content.return_value = {
            "title": "Test Document",
            "content": "Test content",
        }

        self.mock_doc_synth_instance.save_document.return_value = (
            "./test_output/test.json"
        )

        # Call the method
        result = self.client.scrape("https://www.python.org", "Find information")

        # Assert results
        self.assertEqual(result["title"], "Test Document")
        self.assertEqual(result["content"], "Test content")
        self.assertEqual(result["output_path"], "./test_output/test.json")

        # Verify calls
        self.mock_crawler_instance.crawl.assert_called_once_with(
            "https://www.python.org"
        )
        self.mock_ai_processor_instance.analyze_relevance.assert_called_once()
        self.mock_ai_processor_instance.extract_key_info.assert_called_once()
        self.mock_doc_synth_instance.process_extracted_content.assert_called_once()
        self.mock_doc_synth_instance.save_document.assert_called_once()

    def test_batch_scrape(self):
        """Test the batch_scrape method."""
        # Configure mocks for the test
        self.mock_crawler_instance.crawl.return_value = [
            ("https://www.python.org", "<html>Test</html>", "Test content")
        ]

        self.mock_ai_processor_instance.analyze_relevance.return_value = (
            0.8,
            "Relevant",
        )
        self.mock_ai_processor_instance.extract_key_info.return_value = {"key": "value"}

        self.mock_doc_synth_instance.process_extracted_content.return_value = {
            "title": "Test Document",
            "content": "Test content",
        }

        self.mock_doc_synth_instance.save_document.return_value = (
            "./test_output/test.json"
        )

        # Call the method
        results = self.client.batch_scrape(
            ["https://www.python.org", "https://example.org"], "Find information"
        )

        # Assert results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["title"], "Test Document")
        self.assertEqual(results[0]["content"], "Test content")

        # Verify calls
        self.assertEqual(self.mock_crawler_instance.crawl.call_count, 2)
        self.assertEqual(
            self.mock_ai_processor_instance.analyze_relevance.call_count, 2
        )
        self.assertEqual(self.mock_ai_processor_instance.extract_key_info.call_count, 2)
        self.assertEqual(
            self.mock_doc_synth_instance.process_extracted_content.call_count, 2
        )
        self.assertEqual(self.mock_doc_synth_instance.save_document.call_count, 2)


if __name__ == "__main__":
    unittest.main()
