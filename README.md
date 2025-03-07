# AI Text Processing Microservice

A production-ready FastAPI microservice that integrates with Hugging Face Transformers for text processing and summarization in real-time.

## Features

- **Query Processing Endpoint**: Accepts a user's query and returns a structured response
- **Text Summarization Endpoint**: Takes long-form text input and returns a concise summary using DistilBART
- **Asynchronous Processing**: Utilizes FastAPI's async capabilities for high performance
- **Production-Ready**: Includes error handling, logging, input validation, and containerization

## Architecture

The microservice is built with the following architecture:

```
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│                 │        │                 │        │                 │
│  HTTP Requests  │───────▶│  FastAPI App    │───────▶│  Hugging Face   │
│                 │        │                 │        │  Transformers   │
└─────────────────┘        └─────────────────┘        └─────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │                 │
                           │   Async Task    │
                           │   Processing    │
                           │                 │
                           └─────────────────┘
```

### Design Decisions

1. **FastAPI Framework**: Chosen for its high performance, async support, automatic OpenAPI documentation, and built-in validation
2. **Hugging Face Integration**: Uses DistilBART model for efficient text summarization with a good balance of quality and speed
3. **Asynchronous Processing**: Ensures the API remains responsive even during model inference
4. **Thread Pool Executor**: Prevents model inference from blocking the async event loop
5. **Lazy Loading**: Improves startup time by loading models only when needed
6. **Containerization**: Docker-based deployment for consistent environments and easy scaling
7. **Input Validation**: Pydantic models with validators ensure data integrity
8. **Error Handling**: Comprehensive try/except blocks with appropriate HTTP status codes
9. **Logging**: Structured logging for easier debugging and monitoring

## Prerequisites

- Python 3.8+
- pip or conda

## Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-microservice.git
   cd ai-microservice
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the microservice:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t ai-microservice .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 ai-microservice
   ```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PORT` | Port for the FastAPI server | `8000` |
| `LOG_LEVEL` | Logging level (INFO, DEBUG, ERROR) | `INFO` |
| `MODEL_NAME` | Hugging Face model name for summarization | `sshleifer/distilbart-cnn-12-6` |
| `MAX_WORKERS` | Maximum number of worker threads | `3` |

## API Documentation

### Endpoints

#### 1. Root Endpoint
- **URL**: `/`
- **Method**: `GET`
- **Response**: Welcome message and link to API documentation

#### 2. Query Processing
- **URL**: `/query`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "query": "What is artificial intelligence?"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Query processed successfully",
    "query_received": "What is artificial intelligence?"
  }
  ```

#### 3. Text Summarization
- **URL**: `/summarize`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Long text to summarize...",
    "max_length": 150,
    "min_length": 30
  }
  ```
- **Response**:
  ```json
  {
    "original_length": 1024,
    "summary_length": 142,
    "summary": "Concise summary of the text..."
  }
  ```

## Testing

This project includes comprehensive unit tests. To run the tests:

```bash
pytest
```

To run tests with coverage report:

```bash
pytest --cov=main
```

## Security Considerations

- All user inputs are validated using Pydantic models
- Error messages provide minimal information to prevent information leakage
- Rate limiting should be implemented at the infrastructure level
- Consider adding authentication for production deployments

## Performance Optimizations

- The summarization model is optimized for inference speed
- Asynchronous processing ensures high throughput
- Thread pooling prevents blocking the event loop
- Input text is truncated to model context limits to prevent errors

## License

This project is licensed under the MIT License - see the LICENSE file for details.
