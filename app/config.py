from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()

llm_api_key = os.getenv("LLM_API_KEY")
pinecone_vector_db_key = os.getenv("PINECONE_VECTOR_DB_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
llm_name = os.getenv("LLM_NAME")
top_k_retreival = os.getenv("TOP_K_RETRIEVAL")
embedding_model = os.getenv("EMBEDDING_MODEL")
pinecone_name_space = os.getenv("PINECONE_NAME_SPACE")

class Settings(BaseSettings):
    LLM_API_KEY: str = llm_api_key
    PINECONE_VECTOR_DB_KEY: str = pinecone_vector_db_key
    PINECONE_INDEX_NAME: str = pinecone_index_name
    LLM_NAME: str = llm_name
    TOP_K_RETREIVAL: int = top_k_retreival
    EMBEDDING_MODEL: str = embedding_model
    PINECONE_NAME_SPACE: str = pinecone_name_space

    class Config:
        env_file = ".env"
        extra = "allow"  # this causes the error when unexpected variables are passed

settings = Settings()