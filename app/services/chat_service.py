from app.config import settings
from app.services.retrieval import RetrievalService
from fastembed import TextEmbedding

# embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

# Load tiny model (~22MB, downloads once)
embedding_model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)

retrieval_service = RetrievalService(embedding_model)

class ChatService:
    def __init__(self, chat_model_client):
        self.chat_model_client = chat_model_client

    async def get_chat_response(self, user_query):
        return await self.question_answer_bot(user_query = user_query)
    
    async def ask_questions(self, prompt):
        response = await self.chat_model_client.generate_content_async(prompt)
        return response.text
    
    async def question_answer_bot(self, user_query):
        _, matches = await retrieval_service.get_top_retrieval(user_query)

        retrieved_formatted_data = []
        for i in range(len(matches)):
            retrieved_formatted_data.append({
            "id":int(matches[i].id),
            "text":matches[i].metadata['doc'],
            "meta": matches[i].metadata
        })

        ranked_matches = retrieval_service.rerank(user_query, retrieved_formatted_data)
        context = "\n\n".join([doc.get('text') for doc in ranked_matches])

        invalid_question_response_prompt = f"""
                If the given question that is delimited with triple quotes is not related with the given 
                context that is delimited with triple quotes or you donot have or find any appropriate answer from the given context,
                then provide a humble and gentle answer of not having the proper answer and make sure you give answer
                without giving the explanation of the question. 
                Only give the pertains of the context or documents for instruct the user
                of which related questions they should ask to you.

                question: ```{user_query}```
                context: ```{context}```
                """
        
        # Provide Valid Question
        base_prompt = f"""
                    Your task is to perform the following actions: 
                        1 - Look for grammatical mistakes and spelling mistakes in the question that is delimited with triple quotes.
                        2 - If there are any spelling mistakes and grammatical mistakes then correct it.
                        3 - After correcting the spelling mistakes and grammatical mistakes, rephrase the question that is delimited with triple quotes in a refined way.
                        4 - Provide the final question after following the above steps in the  below format.
                If the question that is delimited with triple quotes is related with the given context that is delimited with triple quotes,        
                then use the following format to answer:
                <final question>
                Make sure to provide the final question only.

                If the question that is delimited with triple quotes is not related with context that is delimited with triple quotes,
                then make sure you will only provide the response text that is <response not available>.

                question: ```{user_query}```
                context: ```{context}```
                """
        base_prompt_response = await self.ask_questions(base_prompt)

        # After provide valid question
        if base_prompt_response == "<response not available>":
            prompt = invalid_question_response_prompt
        else: 

            prompt = f"""
                    You will get a question delimited with by triple quotes.
    
                    Your task is to extract relevant information from
                    the given context that is delimited by triple quotes.
    
                    your task to give answer on context extraction
                    If the given question is other than related to the given context or
                    you donot have or find any appropriate answer,
                    then provide a humble and gentle answer of not having the proper answer and
                    without giving the explanation of the question.
                    Only give the pertains of the context or documents for instruct the user 
                    of which related questions they should ask to you.
    
                    question: ```{user_query}``` \
                    context: ```{context}```
                """
        return await self.ask_questions(prompt)
