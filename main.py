# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 22:24:24 2025

@author: Aman Jaiswar
"""

# main.py
import logging
import os
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Text Processing Microservice",
    description="A microservice that provides query processing and text summarization capabilities",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


executor = ThreadPoolExecutor(max_workers=3)

summarizer = None
tokenizer = None
model = None

def load_model():
    global summarizer, tokenizer, model
    logger.info("Loading summarization model...")
    
    # Choose a smaller model appropriate for summarization
    model_name = "sshleifer/distilbart-cnn-12-6"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
        logger.info("Summarization model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

# Request models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User query text")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v

class QueryResponse(BaseModel):
    message: str
    query_received: str

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=100, description="Text to summarize")
    max_length: Optional[int] = Field(150, ge=30, le=500, description="Maximum length of summary")
    min_length: Optional[int] = Field(30, ge=10, le=100, description="Minimum length of summary")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v
    
    @validator('min_length')
    def min_length_valid(cls, v, values):
        if 'max_length' in values and v >= values['max_length']:
            raise ValueError('min_length must be less than max_length')
        return v

class SummarizeResponse(BaseModel):
    original_length: int
    summary_length: int
    summary: str

# Endpoints
@app.get("/")
async def root():
    return {"message": "AI Text Processing Microservice. See /docs for API documentation."}

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    logger.info(f"Received query request with {len(request.query)} characters")
    
    try:
        # Simple echo response for the query endpoint
        return QueryResponse(
            message="Query processed successfully",
            query_received=request.query
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def run_summarization(text: str, max_length: int, min_length: int) -> str:
    """Run the summarization model (in a separate thread)"""
    global summarizer
    
    # Ensure model is loaded
    if summarizer is None:
        raise RuntimeError("Summarization model not initialized")
    
    max_input_length = 1024
    
    truncated_text = text[:max_input_length] if len(text) > max_input_length else text
    
    summary = summarizer(
        truncated_text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )
    
    return summary[0]['summary_text']

@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(request: SummarizeRequest, background_tasks: BackgroundTasks):
    logger.info(f"Received summarization request with {len(request.text)} characters")
    
    global summarizer
    if summarizer is None:
        success = load_model()
        if not success:
            raise HTTPException(
                status_code=503, 
                detail="Failed to initialize summarization service. Please try again later."
            )
    
    try:
        summary = await asyncio.get_event_loop().run_in_executor(
            executor,
            run_summarization,
            request.text,
            request.max_length,
            request.min_length
        )
        
        logger.info(f"Generated summary with {len(summary)} characters")
        
        return SummarizeResponse(
            original_length=len(request.text),
            summary_length=len(summary),
            summary=summary
        )
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

# Initialize models on startup
@app.on_event("startup")
def startup_event():
    load_model()
    
@app.on_event("shutdown")
def shutdown_event():
    executor.shutdown(wait=False)
    logger.info("Application shutting down")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)