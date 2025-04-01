"""
RelevanceChecker Module - Checks content relevance using LLM.

This module provides functionality to evaluate if a piece of content is relevant
to a main content or instructions, with support for multiple LLM models.
"""

from typing import Dict, Optional, List, Tuple
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import OpenAI if available
try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class RelevanceChecker:
    """A class to check content relevance using LLM with multi-model support."""

    # Class variable to track the current model
    _model_type = "ollama"

    @classmethod
    def set_model(cls, model_type: str):
        """Set the model type to use for relevance checking."""
        cls._model_type = model_type.lower()

    def __init__(self, model_name: str = None):
        """Initialize the RelevanceChecker with specified LLM model.

        Args:
            model_name (str): Name of the LLM model to use. If None, uses the class model type.
        """
        # Use model_name if provided, otherwise use class model type
        model_type = model_name.lower() if model_name else self._model_type

        # Initialize LLM based on model type
        if model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI libraries not installed. Please install with 'pip install langchain-openai'"
                )

            # Check for OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OpenAI API key not found. Set OPENAI_API_KEY in environment variables."
                )

            # Initialize OpenAI model
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=500,
                openai_api_key=api_key,
            )
        else:
            # Default to Ollama
            self.llm = Ollama(model="tinyllama", callbacks=None)

        # Define the prompt template for relevance checking
        self.relevance_prompt = PromptTemplate(
            input_variables=["main_content", "check_content"],
            template="""
            Task: Evaluate if the content is relevant to the main content/instructions.
            
            Main Content/Instructions:
            {main_content}
            
            Content to Check:
            {check_content}
            
            Please analyze if this content is relevant to the main content/instructions.
            Return a JSON-like response with the following structure:
            {{
                "is_relevant": true/false,
                "relevance_score": float between 0 and 1,
                "reasoning": "Brief explanation of why the content is relevant or not",
                "key_matches": ["List of key matching concepts or topics found"]
            }}
            
            Response:
            """,
        )

    def clean_content(self, content: str) -> str:
        """Clean and truncate the content to a manageable size.

        Args:
            content (str): Raw webpage content

        Returns:
            str: Cleaned and truncated content
        """
        # Remove extra whitespace and newlines
        content = " ".join(content.split())

        # Truncate to ~2000 characters to avoid token limits
        if len(content) > 2000:
            content = content[:1997] + "..."

        return content

    def extract_response_parts(self, llm_response: str) -> Dict:
        """Extract structured data from LLM response.

        Args:
            llm_response (str): Raw response from LLM

        Returns:
            Dict: Structured response data
        """
        try:
            # Find the JSON-like structure in the response
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1
            json_str = llm_response[start_idx:end_idx]

            # Parse the JSON response
            response_data = json.loads(json_str)

            # Ensure all required fields are present
            required_fields = [
                "is_relevant",
                "relevance_score",
                "reasoning",
                "key_matches",
            ]
            for field in required_fields:
                if field not in response_data:
                    response_data[field] = None

            return response_data

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback response if parsing fails
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "reasoning": f"Error parsing LLM response: {str(e)}",
                "key_matches": [],
            }

    @classmethod
    def check_relevance(
        cls,
        main_content: str,
        check_content: str,
        min_relevance_score: float = 0.5,
        model_override: str = None,
    ) -> bool:
        """Check if the content is relevant to the main content/instructions.

        Args:
            main_content (str): The main content or instructions to compare against
            check_content (str): The content to check for relevance
            min_relevance_score (float): Minimum score to consider content relevant
            model_override (str): Optional model to use for this check

        Returns:
            bool: True if content is relevant, False otherwise
        """
        try:
            # Create an instance with the appropriate model
            model_to_use = model_override if model_override else cls._model_type
            checker = cls(model_name=model_to_use)

            # Clean the content
            cleaned_main = checker.clean_content(main_content)
            cleaned_check = checker.clean_content(check_content)

            # Generate the prompt
            prompt = checker.relevance_prompt.format(
                main_content=cleaned_main, check_content=cleaned_check
            )

            # Get LLM response
            response = checker.llm(prompt)

            # Extract structured data from response
            results = checker.extract_response_parts(response)

            # Determine if content meets relevance threshold
            is_relevant = (
                results["is_relevant"]
                and results["relevance_score"] >= min_relevance_score
            )

            return is_relevant

        except Exception as e:
            # Log error but return True to be conservative about content inclusion
            print(f"Error during relevance check: {str(e)}")
            return True

    @classmethod
    def batch_check_relevance(
        cls,
        main_content: str,
        content_list: List[str],
        min_relevance_score: float = 0.5,
        model_override: str = None,
    ) -> List[bool]:
        """Check relevance for multiple content pieces.

        Args:
            main_content (str): The main content or instructions to compare against
            content_list (List[str]): List of content pieces to check
            min_relevance_score (float): Minimum score to consider content relevant
            model_override (str): Optional model to use for this batch check

        Returns:
            List[bool]: List of relevance results
        """
        return [
            cls.check_relevance(
                main_content, content, min_relevance_score, model_override
            )
            for content in content_list
        ]


# Example usage
if __name__ == "__main__":
    # Test with both models
    main_text = "Information about Python programming language features and libraries"
    check_text = "Python is a high-level, interpreted programming language known for its readability and versatility."

    # Set model to Ollama (default)
    RelevanceChecker.set_model("ollama")
    ollama_result = RelevanceChecker.check_relevance(main_text, check_text)
    print(f"Using Ollama - Content is relevant: {ollama_result}")

    # Try with OpenAI if API key is available
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key and OPENAI_AVAILABLE:
        RelevanceChecker.set_model("openai")
        openai_result = RelevanceChecker.check_relevance(main_text, check_text)
        print(f"Using OpenAI - Content is relevant: {openai_result}")
    else:
        print(
            "OpenAI API key not found or langchain-openai not installed. Skipping OpenAI test."
        )
