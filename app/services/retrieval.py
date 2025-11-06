from app.config import settings
from pinecone import Pinecone
from flashrank import RerankRequest
from flashrank import Ranker

class RetrievalService:
    def __init__(self, embedder):
        self.embedder = embedder

    async def get_embeddings(self, texts):
        return self.embedder.encode(texts)
    
    def pinecone_index_details(self):
        pinecone = Pinecone(api_key=settings.PINECONE_VECTOR_DB_KEY)
        index = pinecone.Index(settings.PINECONE_INDEX_NAME)
        return index

    async def get_top_retrieval(self, user_query):
        # Query Pinecone index
        user_query_vector = await self.get_embeddings(user_query)

        # ✅ Ensure it's a plain list (not numpy array or torch tensor)
        if not isinstance(user_query_vector, list):
            user_query_vector = user_query_vector.tolist()

        index = self.pinecone_index_details()
        response = index.query(
            vector=user_query_vector,
            top_k=int(settings.TOP_K_RETREIVAL),
            include_values=True,
            include_metadata=True,
            namespace=settings.PINECONE_NAME_SPACE
        )
        return user_query_vector, response['matches']

    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./cache")
    def rerank(self, user_query, retrieved_data):
        rerankrequest = RerankRequest(query=user_query, passages=retrieved_data)
        return self.ranker.rerank(rerankrequest)   