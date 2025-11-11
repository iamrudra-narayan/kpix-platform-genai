from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from app.services.chat_service import ChatService
import google.generativeai as genai
from app.config import settings

router = APIRouter()

genai.configure(api_key=settings.LLM_API_KEY)
chat_model_client = genai.GenerativeModel(
            model_name=settings.LLM_NAME,
            system_instruction="You are a helpful research assistant."
        )

chat_service = ChatService(chat_model_client)

class ChatRequest(BaseModel):
    user_prompt: str

    @field_validator('user_prompt')
    def validate_length(cls, v):
        if len(v) > 1000:
            raise ValueError(f'user_prompt too long: {len(v)} characters (max 1000)')
        return v

@router.post("/")
async def chat_endpoint(request: ChatRequest):
    try:
        response = await chat_service.get_chat_response(request.user_prompt)
        return JSONResponse(content=response, status_code=status.HTTP_200_OK)
    except ValueError as ve:
        return JSONResponse(content={"error": f"Invalid input: {ve}"}, status_code=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return JSONResponse(content={"error": "An error occurred while processing your request."}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)