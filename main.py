from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import ai, inbody, users
from app.services.prediction_service import load_models

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_models()  # 모델 형식 오류·모델 없음은 배포(시작) 단계에서 실패시킨다
    yield


app = FastAPI(title="Healthcare API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(inbody.router)
app.include_router(ai.router)


@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)