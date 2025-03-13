from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import hashlib
from rufus.helpers.logger import logger  # Using the RufusLogger instance
from rufus.config import config


class VectorDBCreator:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize VectorDB Creator with configuration."""
        try:
            self.chroma_client = chromadb.Client()

            # Update embedding function to use config
            self.embedding_function = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=config.get(
                        "vector_store.embedding_model", "all-MiniLM-L6-v2"
                    )
                )
            )

            # Update text splitter to use config values
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.get("rag.chunk_size", 1000),
                chunk_overlap=config.get("rag.chunk_overlap", 200),
                length_function=len,
            )
            logger.info("VectorDBCreator initialized successfully")
        except Exception as e:
            logger.error("Error initializing VectorDBCreator", exc_info=True)
            raise

    def create_collection(self, collection_name: str) -> chromadb.Collection:
        """Create or get a ChromaDB collection."""
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            logger.info(f"Created new collection: {collection_name}")
            return collection
        except ValueError:
            # Collection already exists
            collection = self.chroma_client.get_collection(
                name=collection_name, embedding_function=self.embedding_function
            )
            logger.info(f"Retrieved existing collection: {collection_name}")
            return collection
        except Exception as e:
            logger.error(
                f"Error creating/getting collection: {collection_name}", exc_info=True
            )
            raise

    def generate_document_id(self, content: str, url: str) -> str:
        """Generate a unique document ID based on content and URL."""
        combined = f"{url}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def process_documents(
        self, documents: List[Dict[str, Any]], collection_name: str
    ) -> None:
        """
        Process documents and store them in ChromaDB.

        Args:
            documents: List of dictionaries containing document content and metadata
            collection_name: Name of the ChromaDB collection to use
        """
        try:
            collection = self.create_collection(collection_name)

            for doc in documents:
                content = doc.get("content", "")
                url = doc.get("url", "")
                metadata = doc.get("metadata", {})

                # Split text into chunks
                chunks = self.text_splitter.split_text(content)
                logger.debug(f"Split document into {len(chunks)} chunks")

                # Process each chunk
                for i, chunk in enumerate(chunks):
                    doc_id = f"{self.generate_document_id(chunk, url)}_{i}"

                    # Add chunk metadata
                    # Enhanced metadata with source tracking
                    chunk_metadata = {
                        **metadata,
                        "url": url,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_type": metadata.get("source_type", "webpage"),
                        "crawl_timestamp": metadata.get("crawl_timestamp", ""),
                        "title": metadata.get("title", ""),
                        "section": metadata.get("section", ""),
                    }

                    # Add to ChromaDB
                    collection.add(
                        documents=[chunk], metadatas=[chunk_metadata], ids=[doc_id]
                    )

                logger.info(f"Processed document from URL: {url}")

            logger.info(
                f"Successfully processed all documents for collection: {collection_name}"
            )

        except Exception as e:
            logger.error(
                f"Error processing documents for collection: {collection_name}",
                exc_info=True,
            )
            raise

    def query_collection(
        self, collection_name: str, query_text: str, n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar documents.

        Args:
            collection_name: Name of the collection to query
            query_text: Text to search for
            n_results: Number of results to return

        Returns:
            List of dictionaries containing matching documents and their metadata
        """
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name, embedding_function=self.embedding_function
            )

            results = collection.query(query_texts=[query_text], n_results=n_results)

            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append(
                    {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                    }
                )

            logger.info(f"Successfully queried collection: {collection_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Error querying collection: {collection_name}", exc_info=True)
            raise

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from ChromaDB."""
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {collection_name}", exc_info=True)
            raise
