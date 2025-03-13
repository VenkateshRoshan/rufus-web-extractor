"""
RAG Handler Module - Handles Retrieval-Augmented Generation operations.

This module provides functionality for creating RAG pipelines with different LLM models.
Supports both Ollama and OpenAI models.
"""

from typing import List, Dict, Any, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import LLM classes
from langchain_community.llms import Ollama

try:
    from langchain_openai import ChatOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv

from rufus.helpers.logger import logger
from rufus.config import config

# Load environment variables
load_dotenv()


class RAGHandler:
    """
    Handles Retrieval-Augmented Generation (RAG) operations.
    Combines vector store retrieval with LLM-based question answering.
    Support for both Ollama and OpenAI models.
    """

    def __init__(
        self,
        embedding_model: str = config.get(
            "vector_store.embedding_model", "all-MiniLM-L6-v2"
        ),
        temperature: float = config.get("model.temperature", 0.7),
        max_tokens: int = config.get("model.max_tokens", 500),
        model: str = config.get("model.name", "ollama"),
    ):
        """
        Initialize RAG handler with specified models and parameters.

        Args:
            embedding_model: Name of the embedding model
            temperature: Temperature for LLM responses
            max_tokens: Maximum tokens in LLM response
            model: LLM model type to use ('ollama' or 'openai')
        """
        self.model_type = model.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        try:
            # Initialize LLM based on model type
            if self.model_type == "openai":
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
                    temperature=temperature,
                    max_tokens=max_tokens,
                    openai_api_key=api_key,
                )
                logger.info("Initialized RAGHandler with OpenAI model")

            else:  # Default to Ollama
                # Initialize Ollama with TinyLlama
                self.llm = Ollama(model="tinyllama", temperature=temperature)
                logger.info("Initialized RAGHandler with Ollama TinyLlama model")

            # Initialize embedding function (same for both model types)
            self.embedding_function = HuggingFaceEmbeddings(model_name=embedding_model)

        except Exception as e:
            logger.error(f"Failed to initialize RAGHandler: {str(e)}", exc_info=True)
            raise

    def setup_retrieval_qa(
        self, collection_name: str, persist_directory: Optional[str] = None
    ) -> RetrievalQA:
        """
        Set up RetrievalQA chain with vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Optional directory for persistent storage

        Returns:
            Configured RetrievalQA chain
        """
        try:
            # Initialize ChromaDB client
            client = chromadb.Client()

            # Get or create collection
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                ),
            )

            # Create vector store
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=persist_directory,
            )

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_kwargs={"k": 5}  # Get top 5 relevant documents
                ),
                return_source_documents=True,
            )

            logger.info(
                f"RetrievalQA chain setup complete for collection: {collection_name} with model: {self.model_type}"
            )
            return qa_chain

        except Exception as e:
            logger.error(
                f"Failed to setup RetrievalQA for collection: {collection_name}",
                exc_info=True,
            )
            raise

    def generate_answer(
        self, query: str, qa_chain: RetrievalQA, custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate answer for a query using RAG.

        Args:
            query: User query string
            qa_chain: Configured RetrievalQA chain
            custom_prompt: Optional custom prompt template

        Returns:
            Dictionary containing answer, sources, and their URLs
        """
        try:
            # Use custom prompt if provided
            if custom_prompt:
                prompt = PromptTemplate(
                    template=custom_prompt, input_variables=["context", "question"]
                )
                qa_chain.combine_documents_chain.llm_chain.prompt = prompt

            # Generate answer
            response = qa_chain({"query": query})

            # Extract unique source URLs
            source_urls = set()
            source_documents = []

            for doc in response["source_documents"]:
                url = doc.metadata.get("url", "")
                if url:
                    source_urls.add(url)
                source_documents.append(
                    {"content": doc.page_content, "metadata": doc.metadata, "url": url}
                )

            # Format response
            result = {
                "answer": response["result"],
                "source_documents": source_documents,
                "source_urls": list(source_urls),
                "model": self.model_type,
            }

            logger.info(
                f"Generated answer for query using {self.model_type} model: {query[:50]}..."
            )
            return result

        except Exception as e:
            logger.error(f"Failed to generate answer for query: {query}", exc_info=True)
            raise

    def batch_generate_answers(
        self,
        queries: List[str],
        qa_chain: RetrievalQA,
        custom_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.

        Args:
            queries: List of query strings
            qa_chain: Configured RetrievalQA chain
            custom_prompt: Optional custom prompt template

        Returns:
            List of dictionaries containing answers and sources
        """
        try:
            results = []
            for query in queries:
                result = self.generate_answer(query, qa_chain, custom_prompt)
                results.append(result)

            logger.info(
                f"Batch processed {len(queries)} queries using {self.model_type} model"
            )
            return results

        except Exception as e:
            logger.error("Failed to process batch queries", exc_info=True)
            raise
