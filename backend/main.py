# backend/main.py
from fastapi import FastAPI
from backend.api import versions, download, extract, train, dataset

app = FastAPI(title="Rug Pull Detection API")

app.include_router(versions.router, prefix="/api")
app.include_router(download.router, prefix="/api")

app.include_router(extract.router, prefix="/api", tags=["training"])
app.include_router(train.router, prefix="/api", tags=["training"])
app.include_router(dataset.router, prefix='/api', tags=['training'])
