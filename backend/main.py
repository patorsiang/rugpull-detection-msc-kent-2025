# backend/main.py
from fastapi import FastAPI
from backend.api import training, dataset, predict, self_learning, gt, unlabeled, system
from backend.api.error_handlers import register_error_handlers
from backend.utils.middleware import RequestIDMiddleware, AccessLogMiddleware
from backend.utils.logger import setup_logging
from backend.utils.openapi_dynamic import attach_dynamic_file_enums

setup_logging()

app = FastAPI(title="Rug Pull Detection API", version="1.0.0")

app.include_router(training.router, prefix="/api", tags=["training"])

app.include_router(dataset.router, prefix="/api", tags=["dataset"])
app.include_router(gt.router, prefix="/api", tags=["dataset"])
app.include_router(unlabeled.router, prefix="/api", tags=["dataset"])

app.include_router(predict.router, prefix="/api", tags=["predict"])

app.include_router(self_learning.router, prefix="/api", tags=["self learning"])

# new system routes
app.include_router(system.router, prefix="/api", tags=["system"])

# error handlers + middleware
register_error_handlers(app)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(AccessLogMiddleware)

attach_dynamic_file_enums(app)

