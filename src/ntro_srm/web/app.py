"""FastAPI application factory and static file serving for NTRO-SRM Web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ntro_srm.web.api import router as api_router
from ntro_srm.web.services.sr_service import SRService


def create_app(workspace_root: Path | None = None, device: str | None = None) -> FastAPI:
    """Create and configure the FastAPI web application."""
    if workspace_root is None:
        # Default: Project root (NTRO-SRM)
        workspace_root = Path(__file__).resolve().parents[3]
    else:
        workspace_root = Path(workspace_root).resolve()

    app = FastAPI(
        title="NTRO-SRM: Sentinel-2 Super-Resolution Mapping",
        version="0.1.0",
        description="Interactive Geospatial Super-Resolution Web Application (NTRO PS 26142)",
    )

    # Allow local CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize shared SR service singleton
    app.state.sr_service = SRService(
        workspace_root=workspace_root,
        device=device,
        model_variant="lite",
    )

    # Mount static assets and templates
    web_dir = workspace_root / "web"
    static_dir = web_dir / "static"
    templates_dir = web_dir / "templates"

    static_dir.mkdir(parents=True, exist_ok=True)
    templates_dir.mkdir(parents=True, exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))

    # Include REST API
    app.include_router(api_router)

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def index(request: Request):
        """Render main interactive geospatial application dashboard."""
        sys_info = app.state.sr_service.get_system_info()
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"sys_info": sys_info},
        )

    return app


# Default app instance for ASGI servers (e.g. uvicorn src.ntro_srm.web.app:app)
app = create_app()
