from google import genai
from google.genai import types
from app.config import settings
from app.services.retrieval import RetrievalService
from fastembed import TextEmbedding

# Load tiny model (~22MB, downloads once)
embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)

retrieval_service = RetrievalService(embedding_model)

class ChatService:
    def __init__(self, chat_model_client):
        self.chat_model_client: genai.Client = chat_model_client

    async def get_chat_response(self, user_query):
        return await self.question_answer_bot(user_query=user_query)
    
    async def ask_questions(self, prompt):
        # JSON config hata diya taaki sirf text aaye
        response = self.chat_model_client.models.generate_content(
            model=settings.LLM_NAME, 
            contents=prompt
        )
        return response.text.strip()
    
    async def question_answer_bot(self, user_query):
        try:
            _, matches = await retrieval_service.get_top_retrieval(user_query)

            retrieved_formatted_data = []
            for i in range(len(matches)):
                retrieved_formatted_data.append({
                "id":int(matches[i].id),
                "text":matches[i].metadata['doc'],
                "meta": matches[i].metadata
            })

            retrieval_service.build_index(retrieved_formatted_data)
            ranked_matches = retrieval_service.rerank(user_query, retrieved_formatted_data)
            context = "\n\n".join([doc.get('text') for doc in ranked_matches])

            # OPTIMIZED: Ekdum simple evaluation taaki LLM hamesha sahi soche
            base_prompt = f"""Context: {context}
Question: {user_query}
Task: Evaluate if the context contains the answer to the question.
- If NO (unrelated): Output exactly '<response not available>'
- If YES (related): Output exactly '<proceed>'"""
            
            base_prompt_response = await self.ask_questions(base_prompt)

            # Agar context mein answer nahi hai, toh turant empty string return karo. 
            # Isse aapke router ka "if not response:" wali condition perfectly trigger hogi!
            if "<response not available>" in base_prompt_response:
                return "" 
            
            # Agar related hai, toh seedha answer do bina filler words ke
            prompt = f"""Context: {context}
Question: {user_query}
Task: Answer the question clearly using ONLY the context.
CRITICAL RULE: NEVER use phrases like "Based on the context", "According to the text", or "The provided context says". Start your answer directly."""
            
            return await self.ask_questions(prompt)

        except Exception as e:
            # Agar RAG ya kisi line mein code phat raha hoga, toh ab terminal mein print hoga.
            # Taaki aapko pata chale ki error kahan hai.
            print(f"DEBUG ERROR in question_answer_bot: {e}")
            return ""