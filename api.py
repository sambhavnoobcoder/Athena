"""
Adaptive Search API - Production REST API

A scalable, production-ready search API that beats commercial solutions
using only free data sources.

Run: python api.py
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import logging
from datetime import datetime
import uvicorn
import os

from search_engine import AdaptiveSearchSystemV43, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Adaptive Search API",
    description="Free, open-source semantic search that beats commercial solutions",
    version="4.3.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global search engine instance (loaded once on startup)
search_engine: Optional[AdaptiveSearchSystemV43] = None


# Request/Response Models
class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")

    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "top_k": 5
            }
        }


class SearchResultResponse(BaseModel):
    """Individual search result"""
    title: str
    url: str
    snippet: str
    score: float
    source: str
    rank: int


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    results: List[SearchResultResponse]
    metadata: Dict

    class Config:
        schema_extra = {
            "example": {
                "query": "What is the capital of France?",
                "results": [
                    {
                        "title": "Paris - Capital of France",
                        "url": "https://en.wikipedia.org/wiki/Paris",
                        "snippet": "Paris is the capital and most populous city of France...",
                        "score": 0.95,
                        "source": "wikipedia",
                        "rank": 1
                    }
                ],
                "metadata": {
                    "query_type": "specific_factual",
                    "answer_type": "location",
                    "total_results": 5,
                    "search_time_ms": 1234,
                    "timestamp": "2025-11-04T00:00:00Z"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    uptime_seconds: float
    engine_loaded: bool
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup"""
    global search_engine

    logger.info("🚀 Starting Adaptive Search API...")
    logger.info("📦 Loading search engine (this may take a moment)...")

    try:
        search_engine = AdaptiveSearchSystemV43()
        logger.info("✅ Search engine loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load search engine: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down Adaptive Search API...")


# Track startup time for health endpoint
startup_time = time.time()


# API Endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint - redirect to docs"""
    return {
        "message": "Adaptive Search API",
        "version": "4.3.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns system status and uptime
    """
    uptime = time.time() - startup_time

    return HealthResponse(
        status="healthy" if search_engine is not None else "unhealthy",
        version="4.3.0",
        uptime_seconds=uptime,
        engine_loaded=search_engine is not None,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform semantic search

    Args:
        request: SearchRequest with query and top_k

    Returns:
        SearchResponse with ranked results

    Raises:
        HTTPException: If search fails
    """
    if search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized"
        )

    start_time = time.time()

    try:
        logger.info(f"🔍 Search request: '{request.query}' (top_k={request.top_k})")

        # Perform search
        results = search_engine.search(request.query, top_k=request.top_k)

        # Analyze query for metadata
        analysis = search_engine.query_analyzer.analyze(request.query)

        # Convert results to response format
        search_results = []
        for idx, result in enumerate(results, 1):
            search_results.append(SearchResultResponse(
                title=result.title,
                url=result.url,
                snippet=result.snippet[:500],  # Limit snippet length
                score=round(result.score, 4),
                source=result.source,
                rank=idx
            ))

        search_time_ms = int((time.time() - start_time) * 1000)

        logger.info(f"✅ Search completed in {search_time_ms}ms ({len(search_results)} results)")

        return SearchResponse(
            query=request.query,
            results=search_results,
            metadata={
                "query_type": analysis['query_type'],
                "answer_type": analysis['answer_type'],
                "best_source": analysis['best_source'],
                "difficulty": analysis['difficulty'],
                "total_results": len(search_results),
                "search_time_ms": search_time_ms,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )


# Main entry point
if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    workers = int(os.getenv("MAX_WORKERS", "1"))  # Use 1 worker to share engine

    logger.info(f"🚀 Starting API server on {host}:{port}")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )
