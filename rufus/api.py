"""
API Module - FastAPI implementation for Rufus Web Extractor.

This module provides REST API endpoints to access the Rufus web extraction functionality
with support for both Ollama and OpenAI models.
"""

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Body,
    BackgroundTasks,
    Header,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Literal
import os
import json
from pydantic import BaseModel, HttpUrl, Field
import time
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv

# Import Rufus components
from rufus.main import RufusExtractor
from rufus.helpers.logger import logger
from rufus.lib.rag.rag import RAGHandler
from rufus.config import config

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Rufus Web Extractor API",
    description="API for intelligent web data extraction and RAG preparation",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create storage directory for processing results
os.makedirs("api_results", exist_ok=True)


# Pydantic models for request/response validation
class ScrapeRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL to scrape")
    instructions: str = Field(
        ..., description="User instructions for content extraction"
    )
    max_depth: Optional[int] = Field(2, description="Maximum crawl depth")
    max_pages: Optional[int] = Field(20, description="Maximum pages to crawl")
    collection_name: Optional[str] = Field(
        None, description="Custom name for vector store collection"
    )
    model: Optional[str] = Field(
        "ollama", description="LLM model to use (ollama or openai)"
    )


class QueryRequest(BaseModel):
    collection_name: str = Field(..., description="Name of the vector store collection")
    query: str = Field(..., description="Query text to search for in the collection")
    custom_prompt: Optional[str] = Field(
        None, description="Custom prompt template for the LLM"
    )
    model: Optional[str] = Field(
        "ollama", description="LLM model to use (ollama or openai)"
    )


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(BaseModel):
    job_id: str
    status: ProcessingStatus
    created_at: str
    completed_at: Optional[str] = None
    url: str
    result_path: Optional[str] = None
    error: Optional[str] = None
    model: Optional[str] = "ollama"
    collection_name: Optional[str] = None


# In-memory job storage (replace with database in production)
jobs = {}


def get_openai_key():
    """Get OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY in environment variables."
        )
    return api_key


def process_website_task(
    job_id: str,
    url: str,
    instructions: str,
    max_depth: int,
    max_pages: int,
    model: str = "ollama",
    collection_name: Optional[str] = None,
):
    """Background task to process a website with specified model."""
    try:
        # Update job status
        jobs[job_id]["status"] = ProcessingStatus.PROCESSING

        # Configure extractor with job-specific settings and model selection
        rufus_extractor = RufusExtractor(
            use_parallel=True,
            max_depth=max_depth,
            max_pages=max_pages,
            relevance_threshold=0.6,
            model=model,
        )

        # Process the website
        logger.info(f"Starting job {job_id} for URL: {url} with model: {model}")
        results = rufus_extractor.process_website(
            url=url, instructions=instructions, collection_name=collection_name
        )

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = f"api_results/job_{job_id}_{timestamp}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        # Update job status
        jobs[job_id]["status"] = ProcessingStatus.COMPLETED
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["result_path"] = result_path
        jobs[job_id]["model"] = model
        jobs[job_id]["collection_name"] = results.get("collection_name")

        logger.info(f"Completed job {job_id}")

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        jobs[job_id]["status"] = ProcessingStatus.FAILED
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()


@app.get("/")
def read_root():
    return {"message": "Welcome to Rufus Web Extractor API", "status": "active"}


@app.post("/scrape", response_model=JobStatus)
async def scrape_website(background_tasks: BackgroundTasks, request: ScrapeRequest):
    """
    Scrape a website and prepare data for RAG.
    This is an asynchronous endpoint that returns a job ID for status tracking.
    """
    # Check for OpenAI API key if using OpenAI model
    if request.model and request.model.lower() == "openai":
        try:
            get_openai_key()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Generate job ID
    job_id = f"job_{int(time.time())}"

    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": ProcessingStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "url": str(request.url),
        "result_path": None,
        "error": None,
        "model": request.model,
        "collection_name": None,
    }

    # Add task to background tasks
    background_tasks.add_task(
        process_website_task,
        job_id=job_id,
        url=str(request.url),
        instructions=request.instructions,
        max_depth=request.max_depth,
        max_pages=request.max_pages,
        model=request.model,
        collection_name=request.collection_name,
    )

    logger.info(
        f"Created job {job_id} for URL: {request.url} with model: {request.model}"
    )
    return JobStatus(**jobs[job_id])


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(**jobs[job_id])


@app.get("/jobs", response_model=List[JobStatus])
def list_jobs():
    """List all processing jobs."""
    return [JobStatus(**job) for job in jobs.values()]


@app.post("/query")
async def query_collection(request: QueryRequest):
    """
    Query a processed collection using RAG.
    """
    try:
        # Check for OpenAI API key if using OpenAI model
        if request.model and request.model.lower() == "openai":
            try:
                get_openai_key()
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Initialize RAG handler with specified model
        rag_handler = RAGHandler(model=request.model)

        # Set up retrieval QA chain
        qa_chain = rag_handler.setup_retrieval_qa(
            collection_name=request.collection_name
        )

        # Generate answer
        response = rag_handler.generate_answer(
            query=request.query, qa_chain=qa_chain, custom_prompt=request.custom_prompt
        )

        return response

    except Exception as e:
        logger.error(f"Error querying collection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections")
def list_collections():
    """List all available vector store collections."""
    try:
        # This is a simplified implementation
        # In production, you'd want to query ChromaDB directly
        collections = []
        output_dir = "output"

        # Look for result files that contain collection info
        if os.path.exists(output_dir):
            for filename in os.listdir(output_dir):
                if filename.startswith("rufus_results_") and filename.endswith(".json"):
                    file_path = os.path.join(output_dir, filename)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            collections.append(
                                {
                                    "collection_name": data.get("collection_name"),
                                    "url": data.get("url"),
                                    "timestamp": data.get("timestamp"),
                                    "document_count": data.get("stats", {}).get(
                                        "documents_processed", 0
                                    ),
                                    "model": data.get("model", "ollama"),
                                }
                            )
                    except Exception as e:
                        logger.error(
                            f"Error reading collection file {file_path}: {str(e)}"
                        )

        # Also check the api_results directory
        if os.path.exists("api_results"):
            for filename in os.listdir("api_results"):
                if filename.startswith("job_") and filename.endswith(".json"):
                    file_path = os.path.join("api_results", filename)
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            # Only add if not already in the list
                            collection_name = data.get("collection_name")
                            if collection_name and not any(
                                c.get("collection_name") == collection_name
                                for c in collections
                            ):
                                collections.append(
                                    {
                                        "collection_name": collection_name,
                                        "url": data.get("url"),
                                        "timestamp": data.get("timestamp"),
                                        "document_count": data.get("stats", {}).get(
                                            "documents_processed", 0
                                        ),
                                        "model": data.get("model", "ollama"),
                                    }
                                )
                    except Exception as e:
                        logger.error(f"Error reading job file {file_path}: {str(e)}")

        # Also include collections from completed jobs
        for job_id, job_data in jobs.items():
            if job_data.get("status") == ProcessingStatus.COMPLETED and job_data.get(
                "collection_name"
            ):
                collection_name = job_data.get("collection_name")
                if not any(
                    c.get("collection_name") == collection_name for c in collections
                ):
                    collections.append(
                        {
                            "collection_name": collection_name,
                            "url": job_data.get("url"),
                            "timestamp": job_data.get("created_at"),
                            "document_count": 0,  # We don't have this info directly from the job
                            "model": job_data.get("model", "ollama"),
                        }
                    )

        return {"collections": collections}

    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collections/{collection_name}")
def delete_collection(collection_name: str):
    """Delete a vector store collection."""
    try:
        # Initialize a RufusExtractor to access its vectordb_creator
        rufus_extractor = RufusExtractor()

        # Delete the collection
        rufus_extractor.vectordb_creator.delete_collection(collection_name)
        return {"message": f"Collection {collection_name} deleted successfully"}

    except Exception as e:
        logger.error(
            f"Error deleting collection {collection_name}: {str(e)}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Health check endpoint."""
    # Check if both Ollama and OpenAI are available
    models_status = {
        "ollama": True,  # Ollama is local, so we assume it's available
        "openai": bool(os.getenv("OPENAI_API_KEY")),
    }

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": models_status,
    }


@app.get("/models")
def available_models():
    """Get information about available models."""
    models = [
        {
            "name": "ollama",
            "description": "Local Ollama model (TinyLlama)",
            "available": True,  # Local model always available
            "default": True,
        },
        {
            "name": "openai",
            "description": "OpenAI GPT model",
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "default": False,
        },
    ]

    return {"models": models}


# Run the server using Uvicorn when executed directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=config.get("api.host", "0.0.0.0"),
        port=config.get("api.port", 8234),
        reload=True,
    )
