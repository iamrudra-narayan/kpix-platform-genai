# KpiX Platform GenAI Chat API

A FastAPI-based Generative AI chat API that leverages Retrieval-Augmented Generation (RAG) to provide context-aware responses using vector search and document retrieval.

## Features

- **Chat Endpoint**: Interactive chat interface powered by Google Generative AI
- **Retrieval-Augmented Generation (RAG)**: Enhances responses with relevant context from a vector database
- **Vector Search**: Uses Pinecone for efficient document retrieval
- **Document Reranking**: Uses BM25Okapi for improved result relevance
- **Embedding Support**: Utilizes FastEmbed for text embeddings
- **CORS Enabled**: Supports cross-origin requests
- **JWT Authentication**: Bearer token authentication for API security
- **OpenAPI Documentation**: Interactive API docs with Swagger UI

## Tech Stack

- **Backend**: FastAPI (Python)
- **AI/ML**:
  - Google Generative AI (Gemini)
  - Pinecone Vector Database
  - FastEmbed for embeddings
  - BM25Okapi for reranking
- **Deployment**: Render
- **Other**: Pydantic for data validation, Uvicorn for ASGI server

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd KempaasGenAI
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with the required environment variables (see Environment Variables section).

5. Run the application:
   ```bash
   uvicorn app.main:app --reload
   ```

The API will be available at `http://localhost:8000`

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
LLM_API_KEY=your_google_generative_ai_api_key
PINECONE_VECTOR_DB_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
LLM_NAME=gemini-1.5-flash  # or other supported model
TOP_K_RETREIVAL=5  # number of documents to retrieve
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2  # or other FastEmbed model
PINECONE_NAME_SPACE=your_namespace
```

## Usage

### API Endpoints

#### POST /api/chat/
Send a chat message and receive an AI-generated response with context from retrieved documents.

**Request Body:**
```json
{
  "user_prompt": "Your question here"
}
```

**Response:**
```json
{
  "response": "AI-generated answer based on retrieved context"
}
```

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/api/chat/" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer your_jwt_token" \
     -d '{"user_prompt": "What is machine learning?"}'
```

### API Documentation

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Deployment

The application is configured for deployment on Render using the `render.yaml` file.

### Deployed Version

The live API is deployed at: https://kpix-platform-genai.onrender.com/docs

## Project Structure

```
KempaasGenAI/
├── app/
│   ├── main.py              # FastAPI application setup
│   ├── config.py            # Configuration and settings
│   ├── routes/
│   │   └── chat_route.py    # Chat API endpoints
│   └── services/
│       ├── chat_service.py  # Chat logic and RAG implementation
│       └── retrieval.py     # Vector retrieval and reranking
├── requirements.txt         # Python dependencies
├── render.yaml             # Render deployment configuration
├── .env                    # Environment variables (create this)
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]