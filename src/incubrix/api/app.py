"""
FastAPI Application Entrypoint for IncuBrix AI Background Music Engine.
"""

from __future__ import annotations
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, JSONResponse

from incubrix.config import get_settings
from incubrix.api.routes import router
from incubrix.errors import IncubrixError, ErrorCode, incubrix_exception_handler


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="IncuBrix AI Background Music Engine",
        description=(
            "Production-oriented API service for AI background music generation with "
            "genuine instrument stem separation, section mood transitions, and narration ducking."
        ),
        version="1.0.0",
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception Handlers
    app.add_exception_handler(IncubrixError, incubrix_exception_handler)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            loc = " -> ".join([str(l) for l in err.get("loc", [])])
            msg = err.get("msg", "Invalid value")
            details.append({"location": loc, "message": msg})
        
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": ErrorCode.VALIDATION_ERROR,
                    "message": "Input validation failed for the submitted request.",
                    "retryable": False,
                    "details": {"validation_errors": details},
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        code = ErrorCode.INTERNAL_SERVER_ERROR
        if exc.status_code == 404:
            code = ErrorCode.JOB_NOT_FOUND
        elif exc.status_code == 401:
            code = ErrorCode.UNAUTHORIZED

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": code,
                    "message": str(exc.detail),
                    "retryable": False,
                    "details": {},
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": ErrorCode.INTERNAL_SERVER_ERROR,
                    "message": f"An unexpected internal error occurred: {str(exc)}",
                    "retryable": True,
                    "details": {"exception_type": type(exc).__name__},
                }
            },
        )

    # Include API routes
    app.include_router(router)

    # Demo UI Route
    demo_path = Path(__file__).resolve().parent.parent / "web" / "templates" / "demo.html"

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    @app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
    async def demo_page():
        if demo_path.exists():
            return HTMLResponse(content=demo_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h2>IncuBrix Demo Interface</h2><p>Please visit /docs for API documentation.</p>")

    # OpenAPI YAML endpoint
    @app.get("/openapi.yaml", include_in_schema=False)
    async def get_openapi_yaml():
        import yaml
        openapi_schema = app.openapi()
        yaml_content = yaml.dump(openapi_schema, sort_keys=False)
        return Response(content=yaml_content, media_type="application/x-yaml")

    return app


app = create_app()
