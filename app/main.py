from fastapi import FastAPI, Depends, Query, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat_route

app = FastAPI(
    title="KpiX Platform GenAI Chat API",
    version="1.0.0",
    description="KpiX Platform GenAI Chat API",
    openapi_tags=[
        {"name": "CHAT"},
    ]
)

# Allowed origins for CORS
# origins = settings.CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Swagger Auth Token support
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Routers
app.include_router(chat_route.router, prefix="/api/chat", tags=["CHAT"])

# ✅ Swagger OpenAPI customization
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Ensure 'components' exists
    if "components" not in openapi_schema or not isinstance(openapi_schema["components"], dict):
        openapi_schema["components"] = {}

    # Ensure securitySchemes exists
    components = openapi_schema["components"]
    if "securitySchemes" not in components or not isinstance(components["securitySchemes"], dict):
        components["securitySchemes"] = {}

    # Define the BearerAuth scheme
    components["securitySchemes"]["BearerAuth"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT"
    }

    # (Optional) apply globally so all endpoints use BearerAuth unless overridden
    # This sets a global security requirement. Remove if you don't want global enforcement.
    openapi_schema.setdefault("security", [])
    # only add if not already present
    if {"BearerAuth": []} not in openapi_schema["security"]:
        openapi_schema["security"].append({"BearerAuth": []})

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi