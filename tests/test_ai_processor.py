"""
Unit tests for the AI processor module.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rufus_2.ai_processor import AIProcessor


class TestAIProcessor(unittest.TestCase):
    """Tests for the AIProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock AI processor that doesn't make real API calls
        self.patcher = patch("rufus.ai_processor.AIProcessor._call_ai_api")
        self.mock_call_ai = self.patcher.start()
        self.processor = AIProcessor(api_key="dummy_key")

    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher.stop()

    def test_analyze_relevance(self):
        """Test the analyze_relevance method."""
        # Set up the mock to return a specific response
        self.mock_call_ai.return_value = json.dumps(
            {
                "score": 0.75,
                "justification": "The content is relevant because it discusses Python programming.",
            }
        )

        # Call the method
        score, justification = self.processor.analyze_relevance(
            "https://www.python.org",
            "Python is a programming language",
            "Find information about Python",
        )

        # Assert results
        self.assertEqual(score, 0.75)
        self.assertEqual(
            justification,
            "The content is relevant because it discusses Python programming.",
        )
        self.mock_call_ai.assert_called_once()

    def test_extract_key_info(self):
        """Test the extract_key_info method."""
        # Set up the mock to return a specific response
        self.mock_call_ai.return_value = json.dumps(
            {
                "language": "Python",
                "description": "A high-level programming language",
                "website": "python.org",
            }
        )

        # Call the method
        result = self.processor.extract_key_info(
            "https://www.python.org",
            "Python is a programming language",
            "Extract information about Python",
        )

        # Assert results
        self.assertEqual(result["language"], "Python")
        self.assertEqual(result["description"], "A high-level programming language")
        self.assertEqual(result["website"], "python.org")
        self.mock_call_ai.assert_called_once()

    def test_synthesize_document(self):
        """Test the synthesize_document method."""
        # Set up the mock to return a specific response
        self.mock_call_ai.return_value = json.dumps(
            {
                "title": "Python Programming Language",
                "sections": [
                    {
                        "heading": "Introduction",
                        "content": "Python is a programming language.",
                    }
                ],
            }
        )

        # Create sample extracted info
        extracted_info = [
            {
                "url": "https://www.python.org",
                "relevance": 0.8,
                "language": "Python",
                "description": "A programming language",
            }
        ]

        # Call the method
        result = self.processor.synthesize_document(
            extracted_info, "Create a document about Python"
        )

        # Assert results
        self.assertEqual(result["title"], "Python Programming Language")
        self.assertEqual(len(result["sections"]), 1)
        self.assertEqual(result["sections"][0]["heading"], "Introduction")
        self.mock_call_ai.assert_called_once()


if __name__ == "__main__":
    unittest.main()
