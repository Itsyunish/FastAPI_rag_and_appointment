from fastapi import FastAPI
from practise.routes import (
    file_routes,
    query_routes,
    appointment_routes,
    redis_routes,
    mongo_routes
)
from practise.utils.file_utils import ensure_upload_dir
from practise.config import UPLOAD_DIR

ensure_upload_dir(UPLOAD_DIR)

app = FastAPI(title="RAG API")

# Include routers
app.include_router(file_routes.router, prefix="/files", tags=["Files"])
app.include_router(query_routes.router, prefix="/rag", tags=["RAG Queries"])
app.include_router(appointment_routes.router, prefix="/appointments", tags=["Appointments"])
app.include_router(redis_routes.router, prefix="/redis", tags=["Redis"])
app.include_router(mongo_routes.router, prefix="/mongo", tags=["MongoDB"])
