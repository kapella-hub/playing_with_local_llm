"""FastAPI application with document upload and question answering endpoints."""
import logging
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.ingestion import ingest_files, ingest_url, refresh_url_content
from app.vectorstore import query_top_k, get_collection_stats, clear_collection, list_files, delete_file_by_source
from app.llm import generate_answer

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Local RAG System",
    description="Document ingestion and question answering using local LLM and ChromaDB",
    version="1.0.0"
)

# Mount static files directory
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Request/Response Models
class UploadResponse(BaseModel):
    """Response model for document upload endpoint."""
    batch_id: str = Field(..., description="Unique batch identifier")
    file_count: int = Field(..., description="Number of files processed")
    success_count: int = Field(..., description="Number of successfully ingested files")
    failed_count: int = Field(..., description="Number of failed files")
    total_chunks: int = Field(..., description="Total number of chunks created")
    failed_files: List[dict] = Field(default_factory=list, description="List of failed files with errors")


class AskRequest(BaseModel):
    """Request model for question answering endpoint."""
    query: str = Field(..., description="Question to answer")
    top_k: Optional[int] = Field(None, description="Number of chunks to retrieve (optional)")


class AskResponse(BaseModel):
    """Response model for question answering endpoint."""
    answer: str = Field(..., description="Generated answer")


class StatsResponse(BaseModel):
    """Response model for collection statistics."""
    collection_name: str
    document_count: int


class ClearResponse(BaseModel):
    """Response model for clear operation."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    collection_name: str = Field(..., description="Name of the cleared collection")


class FileInfo(BaseModel):
    """Model for file information."""
    filename: str = Field(..., description="File name")
    upload_date: str = Field(..., description="Upload timestamp (ISO format)")
    chunk_count: int = Field(..., description="Number of chunks for this file")


class FileListResponse(BaseModel):
    """Response model for file list endpoint."""
    files: List[FileInfo] = Field(..., description="List of files")
    total_files: int = Field(..., description="Total number of files")


class DeleteFileResponse(BaseModel):
    """Response model for file deletion."""
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")
    deleted_count: int = Field(..., description="Number of chunks deleted")


class UploadUrlRequest(BaseModel):
    """Request model for URL upload endpoint."""
    url: str = Field(..., description="URL to fetch and ingest")
    follow_links: bool = Field(default=False, description="Whether to follow and ingest links found on the page")
    max_depth: int = Field(default=1, ge=1, le=5, description="Maximum crawl depth (1-5). Only used when follow_links=True")
    same_domain_only: bool = Field(default=True, description="Only follow links within the same domain. Only used when follow_links=True")


class UploadUrlResponse(BaseModel):
    """Response model for URL upload endpoint."""
    success: bool = Field(..., description="Whether ingestion was successful")
    source_url: str = Field(..., description="The source URL")
    total_chunks: int = Field(..., description="Number of chunks created")
    urls_processed: int = Field(default=0, description="Number of URLs successfully processed")
    urls_failed: int = Field(default=0, description="Number of URLs that failed to process")
    last_fetched: Optional[str] = Field(None, description="ISO timestamp of when content was fetched")
    error: Optional[str] = Field(None, description="Error message if failed")


class RefreshUrlResponse(BaseModel):
    """Response model for URL refresh endpoint."""
    success: bool = Field(..., description="Whether refresh was successful")
    source_url: str = Field(..., description="The source URL")
    total_chunks: int = Field(..., description="Number of chunks created")
    last_fetched: Optional[str] = Field(None, description="ISO timestamp of when content was fetched")
    error: Optional[str] = Field(None, description="Error message if failed")


# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based solely on the provided context.

Rules:
1. Answer ONLY using information from the context provided below
2. **CRITICAL: Always include URLs, web addresses, access links, and contact information when present in the context**
   - URLs are extremely important and must be included prominently in your answer
   - If asking about a tool/system, prioritize its URL/access link in the first paragraph
3. If the answer cannot be found in the context, respond with "I don't know based on the provided documents"
4. ALWAYS format your answer using markdown for better readability:
   - Use **bold** for important terms, tool names, URLs, and key concepts
   - Break down information into bullet points (-) when listing features, requirements, or multiple items
   - Use numbered lists (1., 2., 3.) for sequential steps or procedures
   - Add blank lines between different sections or topics
   - Structure your response clearly with proper paragraphs
5. Do not make up information or use external knowledge
6. If the context is insufficient, acknowledge the limitation

Example of good formatting:
**ToolName** is a system for XYZ, accessible at **https://example.com**

**Key Features:**
- Feature 1 with description
- Feature 2 with description
- Feature 3 with description

**Access Requirements:**
- Requirement 1
- Requirement 2"""


@app.get("/")
def root():
    """Serve the web UI."""
    html_file = Path(__file__).parent.parent / "static" / "index.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        return {
            "name": "Local RAG System",
            "version": "1.0.0",
            "message": "Web UI not found. API endpoints available at /docs",
            "endpoints": {
                "upload": "/documents/upload",
                "ask": "/ask",
                "stats": "/stats"
            }
        }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Get collection statistics."""
    try:
        stats = get_collection_stats()
        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/manage")
def serve_manage_page():
    """Serve the document management page."""
    html_file = Path(__file__).parent.parent / "static" / "manage.html"
    if html_file.exists():
        return FileResponse(html_file)
    else:
        raise HTTPException(status_code=404, detail="Manage page not found")


@app.get("/documents/list", response_model=FileListResponse)
def list_all_files():
    """List all uploaded files with metadata."""
    try:
        logger.info("Listing all files")
        files = list_files()
        return FileListResponse(
            files=files,
            total_files=len(files)
        )
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/file/{filename}", response_model=DeleteFileResponse)
def delete_file(filename: str):
    """Delete a specific file and all its chunks."""
    try:
        logger.info(f"Deleting file: {filename}")
        result = delete_file_by_source(filename)
        
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        
        logger.info(f"Successfully deleted file: {filename}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/clear", response_model=ClearResponse)
def clear_all_documents():
    """Clear all documents from the vector store."""
    try:
        logger.info("Clearing all documents from vector store")
        result = clear_collection()
        logger.info(f"Successfully cleared all documents: {result}")
        return result
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload-url", response_model=UploadUrlResponse)
def upload_url(request: UploadUrlRequest):
    """
    Fetch and ingest content from a URL.
    
    This endpoint will:
    1. Fetch content from the provided URL
    2. Extract text from the HTML
    3. Optionally crawl and ingest linked pages (if follow_links=True)
    4. Chunk the text
    5. Add it to the vector store with URL metadata
    
    Parameters:
    - url: The starting URL to ingest
    - follow_links: If True, crawl and ingest linked pages (default: False)
    - max_depth: Maximum crawl depth, 1-5 (default: 2). Only applies when follow_links=True
    - same_domain_only: Only follow links within same domain (default: True). Only applies when follow_links=True
    
    The URL(s) will be stored as sources and can be refreshed later.
    """
    if not request.url or not request.url.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    # Validate URL format
    if not request.url.startswith(('http://', 'https://')):
        raise HTTPException(
            status_code=400, 
            detail="URL must start with http:// or https://"
        )
    
    crawl_info = ""
    if request.follow_links:
        crawl_info = f" (crawling with max_depth={request.max_depth}, same_domain_only={request.same_domain_only})"
    
    logger.info(f"Processing URL upload: {request.url}{crawl_info}")
    
    try:
        # Generate unique document ID for this URL
        doc_id = f"url:{uuid.uuid4()}"
        
        # Ingest the URL with crawling options
        result = ingest_url(
            request.url, 
            doc_id,
            follow_links=request.follow_links,
            max_depth=request.max_depth,
            same_domain_only=request.same_domain_only
        )
        
        if result["success"]:
            logger.info(
                f"Successfully ingested URL {request.url}: "
                f"{result.get('urls_processed', 1)} URL(s) processed, "
                f"{result['total_chunks']} chunks"
            )
        else:
            logger.warning(f"Failed to ingest URL {request.url}: {result.get('error')}")
        
        return UploadUrlResponse(**result)
        
    except Exception as e:
        logger.error(f"Error during URL ingestion: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to ingest URL: {str(e)}"
        )


@app.post("/documents/refresh-url/{filename:path}", response_model=RefreshUrlResponse)
def refresh_url(filename: str):
    """
    Refresh content from a previously ingested URL.
    
    This endpoint will:
    1. Delete existing chunks for the URL
    2. Re-fetch content from the URL
    3. Re-chunk and re-index the new content
    
    Args:
        filename: The URL (as stored in the source field)
    """
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="URL cannot be empty")
    
    logger.info(f"Refreshing URL: {filename}")
    
    try:
        # Generate new document ID for the refresh
        doc_id = f"url:{uuid.uuid4()}"
        
        # Refresh the URL content
        result = refresh_url_content(filename, doc_id)
        
        if result["success"]:
            logger.info(
                f"Successfully refreshed URL {filename}: "
                f"{result['total_chunks']} chunks"
            )
        else:
            logger.warning(f"Failed to refresh URL {filename}: {result.get('error')}")
        
        return RefreshUrlResponse(**result)
        
    except Exception as e:
        logger.error(f"Error refreshing URL {filename}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh URL: {str(e)}"
        )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and ingest multiple documents.
    
    Accepts multiple files in various formats:
    - Text: .txt, .md
    - PDF: .pdf
    - Office: .docx, .pptx, .xls, .xlsx
    - Web: .html, .htm
    - Data: .csv
    - Images: .png, .jpg, .jpeg, .tif, .tiff (OCR)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Generate unique batch ID
    batch_id = str(uuid.uuid4())
    
    # Create batch directory
    batch_dir = Path(settings.upload_dir) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing upload batch {batch_id} with {len(files)} files")
    
    # Save uploaded files
    saved_paths = []
    for file in files:
        try:
            file_path = batch_dir / file.filename
            
            # Save file
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            saved_paths.append(file_path)
            logger.info(f"Saved file: {file.filename}")
            
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
            # Continue with other files
    
    if not saved_paths:
        raise HTTPException(status_code=500, detail="Failed to save any files")
    
    # Ingest files
    try:
        result = ingest_files(saved_paths, batch_id)
        
        logger.info(
            f"Batch {batch_id} ingestion complete: "
            f"{result['success_count']} success, "
            f"{result['failed_count']} failed, "
            f"{result['total_chunks']} chunks"
        )
        
        return UploadResponse(
            batch_id=batch_id,
            file_count=len(files),
            success_count=result["success_count"],
            failed_count=result["failed_count"],
            total_chunks=result["total_chunks"],
            failed_files=result["failed_files"]
        )
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Answer a question using RAG over ingested documents.
    
    The endpoint:
    1. Retrieves relevant document chunks from the vector store
    2. Constructs a prompt with context and question
    3. Calls the local LLM to generate an answer
    4. Returns ONLY the answer (no context or internal details)
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Processing question: {request.query[:100]}...")
    
    try:
        # Retrieve relevant chunks
        top_k = request.top_k if request.top_k is not None else settings.top_k
        results = query_top_k(request.query, top_k=top_k)
        
        if not results:
            logger.warning("No relevant documents found")
            return AskResponse(
                answer="I don't have any documents to answer this question. Please upload documents first."
            )
        
        # Build context from retrieved chunks
        context_parts = []
        for idx, result in enumerate(results, 1):
            context_parts.append(f"[Document {idx}]\n{result['text']}")
        
        context_str = "\n\n".join(context_parts)
        
        # Build user prompt with context and question
        user_prompt = f"""Context from documents:

{context_str}

Question: {request.query}

Answer:"""
        
        # Generate answer using local LLM
        logger.info("Generating answer with local LLM...")
        answer = generate_answer(user_prompt, system_prompt=SYSTEM_PROMPT)
        
        logger.info(f"Answer generated: {answer[:100]}...")
        
        return AskResponse(answer=answer)
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process question: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.api:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True
    )
