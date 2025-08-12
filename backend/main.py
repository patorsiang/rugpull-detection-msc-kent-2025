# backend/main.py
from fastapi import FastAPI
from backend.api import versions, train, dataset, predict, self_learning
from backend.utils.logger import setup_logging

setup_logging()

app = FastAPI(title="Rug Pull Detection API")

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI in Docker!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

app.include_router(versions.router, prefix="/api")

app.include_router(train.router, prefix="/api", tags=["training"])

app.include_router(dataset.router, prefix="/api", tags=["dataset"])

app.include_router(predict.router, prefix="/api", tags=["predict"])

app.include_router(self_learning.router, prefix="/api", tags=["self learning"])
