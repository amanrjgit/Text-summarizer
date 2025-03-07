# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 21:31:35 2025

@author: Aman Jaiswar
"""

# test_main.py
import pytest
from fastapi.testclient import TestClient
import os
from unittest.mock import patch, MagicMock

from main import app

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_transformer_pipeline():
    with patch("main.pipeline") as mock_pipeline:
        mock_summarizer = MagicMock()
        mock_summarizer.return_value = [{"summary_text": "This is a mock summary."}]
        mock_pipeline.return_value = mock_summarizer
        yield mock_pipeline


@pytest.fixture(autouse=True)
def mock_model_loading():
    with patch("main.AutoTokenizer") as mock_tokenizer, \
         patch("main.AutoModelForSeq2SeqLM") as mock_model:
        mock_tokenizer.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value = MagicMock()
        yield

def test_root_endpoint():
    """Test the root endpoint returns the expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "AI Text Processing Microservice" in response.json()["message"]

def test_query_endpoint_valid_input():
    """Test the query endpoint with valid input."""
    response = client.post(
        "/query",
        json={"query": "What is artificial intelligence?"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "message": "Query processed successfully",
        "query_received": "What is artificial intelligence?"
    }

def test_query_endpoint_empty_input():
    """Test the query endpoint with empty input."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422

def test_query_endpoint_too_long_input():
    """Test the query endpoint with input exceeding max length."""
    response = client.post(
        "/query",
        json={"query": "x" * 1001}
    )
    assert response.status_code == 422

def test_summarize_endpoint_valid_input():
    """Test the summarize endpoint with valid input."""
    with patch("main.summarizer") as mock_summarizer:
        mock_summarizer.return_value = [{"summary_text": "This is a mock summary."}]
        
        response = client.post(
            "/summarize",
            json={
                "text": "This is a long text that needs to be summarized. " * 10,
                "max_length": 150,
                "min_length": 30
            }
        )
        assert response.status_code == 200
        assert "summary" in response.json()
        assert response.json()["summary"] == "This is a mock summary."
        assert "original_length" in response.json()
        assert "summary_length" in response.json()

def test_summarize_endpoint_too_short_input():
    """Test the summarize endpoint with input that's too short."""
    response = client.post(
        "/summarize",
        json={
            "text": "Too short.",
            "max_length": 150,
            "min_length": 30
        }
    )
    assert response.status_code == 422

def test_summarize_endpoint_invalid_lengths():
    """Test the summarize endpoint with invalid min/max lengths."""
    response = client.post(
        "/summarize",
        json={
            "text": "This is a long text that needs to be summarized. " * 10,
            "max_length": 50,
            "min_length": 60
        }
    )
    assert response.status_code == 422
        
def test_summarize_endpoint_no_model():
    """Test the summarize endpoint when the model is not loaded."""
    with patch("main.load_model", return_value=False):
        with patch("main.summarizer", None):
            response = client.post(
                "/summarize",
                json={
                    "text": "This is a long text that needs to be summarized. " * 10,
                    "max_length": 150,
                    "min_length": 30
                }
            )
            assert response.status_code == 503
