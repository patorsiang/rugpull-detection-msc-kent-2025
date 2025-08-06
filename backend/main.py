# backend/main.py
from fastapi import FastAPI
from backend.api import versions, download  # Import the router

app = FastAPI(title="Rug Pull Detection API")

app.include_router(versions.router, prefix="/api")
app.include_router(download.router, prefix="/api")
