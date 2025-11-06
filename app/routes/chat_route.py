from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.services.chat_service import ChatService
from openai import OpenAI
from app.config import settings

router = APIRouter()

openai_client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=settings.LLM_API_KEY,
    )

chat_service = ChatService(openai_client)

@router.post("/")
async def chat_endpoint(user_prompt: str):
    response = await chat_service.get_chat_response(user_prompt)
    return JSONResponse(content={"message": response}, status_code=status.HTTP_200_OK)