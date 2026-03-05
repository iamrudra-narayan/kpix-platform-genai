from app.config import settings
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

class RetrievalService:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.bm25 = None
    
    async def get_embeddings(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        
        # Convert generator → list of np arrays → list of lists
        embeddings = [emb.tolist() for emb in self.embedding_model.embed(texts)]
        return embeddings[0] if len(embeddings) == 1 else embeddings
    
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

    def build_index(self, retrieved_data):
        self.bm25 = BM25Okapi(retrieved_data)

    def rerank(self, user_query, retrieved_data):
        if not self.bm25:
            raise ValueError("BM25 index not built – call build_index() first")

        # Use only top-k (already filtered by Pinecone)
        hits = retrieved_data[:int(settings.TOP_K_RETREIVAL)]

        # Score the *same* docs that are in the index
        query_tokens = user_query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Attach scores (order matches because retrieved_data order = index order)
        for hit, score in zip(hits, scores):
            hit["final_score"] = score

        return sorted(hits, key=lambda x: x["final_score"], reverse=True)  