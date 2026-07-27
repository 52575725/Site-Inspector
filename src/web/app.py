from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from config.settings import Settings
from src.storage.database import get_session_factory, init_db
from src.web.security import is_allowed_web_origin
from src.web.routes.article_images import router as article_images_router
from src.web.routes.articles import router as articles_router
from src.web.routes.dashboard import router as dashboard_router
from src.web.routes.fixes import router as fixes_router
from src.web.routes.fix_actions import router as fix_actions_router
from src.web.routes.issues import router as issues_router
from src.web.routes.scans import router as scans_router
from src.web.routes.settings_api import router as settings_api_router
from src.web.routes.tools import router as tools_router

BASE_DIR = Path(__file__).resolve().parent


def create_app() -> FastAPI:
    settings = Settings.load()
    app = FastAPI(title="Site Inspector Dashboard", version="0.1.0")
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["127.0.0.1", "localhost", "testserver"],
    )

    @app.middleware("http")
    async def protect_local_mutations(request, call_next):
        if request.url.path.startswith("/api/") and request.method in {
            "POST", "PUT", "PATCH", "DELETE",
        }:
            origin = request.headers.get("origin")
            if origin and not is_allowed_web_origin(origin):
                return JSONResponse(
                    {"detail": "Cross-origin API mutations are not allowed"},
                    status_code=403,
                )
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    app.include_router(article_images_router)
    app.include_router(articles_router)
    app.include_router(dashboard_router)
    app.include_router(issues_router)
    app.include_router(fix_actions_router)
    app.include_router(fixes_router)
    app.include_router(scans_router)
    app.include_router(settings_api_router)
    app.include_router(tools_router)

    @app.on_event("startup")
    async def startup():
        await init_db(settings)
        factory = get_session_factory(settings)
        app.state.session_factory = factory
        app.state.settings = settings

    @app.on_event("shutdown")
    async def shutdown():
        pass

    return app
