"""
FastAPI Web Application Entrypoint for CardioGuard AI.
Configures CORS, static assets, Jinja2 templates, API routers, and main dashboard view.
"""

import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.routes.predict import router as predict_router
from app.routes.metrics import router as metrics_router
from app.routes.health import router as health_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Enable CORS for cross-origin integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Template directories
static_dir = os.path.join(os.path.dirname(__file__), "static")
templates_dir = os.path.join(os.path.dirname(__file__), "templates")

os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Include API Routers under /api/v1 prefix
app.include_router(predict_router, prefix=settings.API_V1_STR)
app.include_router(metrics_router, prefix=settings.API_V1_STR)
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(health_router)  # Also expose /health at root level

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard(request: Request):
    """Renders the main interactive CardioGuard AI web dashboard."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "project_name": settings.PROJECT_NAME,
            "version": settings.VERSION
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main.app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
