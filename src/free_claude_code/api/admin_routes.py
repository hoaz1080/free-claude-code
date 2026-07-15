"""Local admin UI routes and APIs."""

import ipaddress
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from free_claude_code.application.model_metadata import ProviderModelRefreshResult
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.admin.persistence import validate_updates
from free_claude_code.config.admin.values import load_config_response
from free_claude_code.config.custom_providers import (
    CustomProviderDefinition,
    load_custom_providers_from_managed_env,
    save_custom_providers_to_managed_env,
)
from free_claude_code.config.detection import (
    detect_provider_profile,
    generate_provider_id,
)
from free_claude_code.config.model_refs import configured_chat_model_refs

from .dependencies import get_services
from .ports import ApiServices

router = APIRouter()

STATIC_DIR = Path(__file__).resolve().parent / "admin_static"
LOCAL_PROVIDER_PATHS = {
    "lmstudio": "/models",
    "llamacpp": "/models",
    "ollama": "/api/tags",
}


class AdminConfigPayload(BaseModel):
    """Partial config update submitted by the admin UI."""

    values: dict[str, Any] = Field(default_factory=dict)


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _origin_is_local(origin: str | None) -> bool:
    if not origin:
        return True
    parsed = urlsplit(origin)
    return _is_loopback_host(parsed.hostname)


def require_loopback_admin(request: Request) -> None:
    """Allow admin access only from the local machine."""

    client_host = request.client.host if request.client else None
    if not _is_loopback_host(client_host):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")

    origin = request.headers.get("origin")
    if not _origin_is_local(origin):
        raise HTTPException(status_code=403, detail="Admin UI is local-only")


def _asset_response(filename: str) -> FileResponse:
    path = STATIC_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return FileResponse(path)


@router.get("/admin", include_in_schema=False)
async def admin_page(request: Request):
    require_loopback_admin(request)
    return _asset_response("index.html")


@router.get("/admin/assets/{filename}", include_in_schema=False)
async def admin_asset(filename: str, request: Request):
    require_loopback_admin(request)
    if filename not in {"admin.css", "admin.js"}:
        raise HTTPException(status_code=404, detail="Admin asset not found")
    return _asset_response(filename)


@router.get("/admin/api/config")
async def get_admin_config(request: Request):
    require_loopback_admin(request)
    return load_config_response()


@router.post("/admin/api/config/validate")
async def validate_admin_config(payload: AdminConfigPayload, request: Request):
    require_loopback_admin(request)
    return validate_updates(_filtered_values(payload.values))


@router.post("/admin/api/config/apply")
async def apply_admin_config(
    payload: AdminConfigPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.apply_admin_config(_filtered_values(payload.values))
    restart = result.get("restart")
    if isinstance(restart, dict) and restart.get("automatic"):
        background_tasks.add_task(services.admin.request_restart)
    return result


@router.get("/admin/api/status")
async def admin_status(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return services.admin.admin_status()


@router.get("/admin/api/providers/local-status")
async def local_provider_status(request: Request):
    require_loopback_admin(request)
    config = load_config_response()
    values = {field["key"]: field["value"] for field in config["fields"]}
    checks = []
    for provider_id, path in LOCAL_PROVIDER_PATHS.items():
        base_url = _local_provider_url(provider_id, values)
        checks.append(await _check_local_provider(provider_id, base_url, path))
    return {"providers": checks}


@router.post("/admin/api/providers/{provider_id}/test")
async def test_provider(
    provider_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return await services.admin.test_provider(provider_id)


@router.get("/admin/api/models")
async def models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    return _model_options(services)


@router.post("/admin/api/models/refresh")
async def refresh_models(
    request: Request,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    result = await services.admin.refresh_models()
    return _model_options(services, refresh_result=result)


def _model_options(
    services: ApiServices,
    *,
    refresh_result: ProviderModelRefreshResult | None = None,
) -> dict[str, list[str]]:
    configured = {
        ref.model_ref
        for ref in configured_chat_model_refs(services.requests.current_settings())
    }
    discovered = {
        info.model_id for info in services.requests.cached_prefixed_model_infos()
    }
    failed_provider_ids = (
        refresh_result.failed_provider_ids if refresh_result is not None else ()
    )
    return {
        "models": sorted(configured | discovered, key=str.casefold),
        "failed_providers": list(failed_provider_ids),
    }


class CustomProviderPayload(BaseModel):
    """Custom provider definition submitted by the admin UI."""

    base_url: str = Field(min_length=1)
    api_keys: list[str] = Field(min_length=1)
    display_name: str = Field(default="")
    provider_id: str = Field(default="")


def _custom_provider_status(defn: CustomProviderDefinition) -> dict[str, Any]:
    return {
        "provider_id": defn.provider_id,
        "display_name": defn.display_name,
        "base_url": defn.base_url,
        "detected_profile": defn.detected_profile,
        "api_key_count": len(defn.api_keys),
        "kind": "custom",
        "status": "configured",
        "label": f"Custom ({defn.detected_profile or 'generic'})",
    }


@router.get("/admin/api/custom-providers")
async def list_custom_providers(request: Request):
    require_loopback_admin(request)
    definitions = load_custom_providers_from_managed_env()
    return {
        "providers": [_custom_provider_status(defn) for defn in definitions.values()]
    }


@router.post("/admin/api/custom-providers")
async def add_custom_provider(
    payload: CustomProviderPayload,
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    definitions = load_custom_providers_from_managed_env()

    # Auto-detect or use provided provider_id
    provider_id = payload.provider_id.strip() or generate_provider_id(payload.base_url)
    if provider_id in definitions:
        raise HTTPException(
            status_code=409,
            detail=f"Custom provider '{provider_id}' already exists",
        )

    display_name = payload.display_name.strip() or provider_id.replace("_", " ").title()
    detected_profile = detect_provider_profile(payload.base_url)
    api_keys = tuple(k.strip() for k in payload.api_keys if k.strip())
    if not api_keys:
        raise HTTPException(status_code=400, detail="At least one API key is required")

    definition = CustomProviderDefinition(
        provider_id=provider_id,
        display_name=display_name,
        base_url=payload.base_url.strip().rstrip("/"),
        api_keys=api_keys,
        detected_profile=detected_profile,
    )
    definitions[provider_id] = definition
    save_custom_providers_to_managed_env(definitions)

    # Trigger provider generation refresh so the new provider takes effect
    await services.admin.refresh_catalog()
    background_tasks.add_task(services.admin.request_restart)

    return {
        "ok": True,
        "provider": _custom_provider_status(definition),
    }


@router.delete("/admin/api/custom-providers/{provider_id}")
async def delete_custom_provider(
    provider_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    services: ApiServices = Depends(get_services),
):
    require_loopback_admin(request)
    definitions = load_custom_providers_from_managed_env()
    if provider_id not in definitions:
        raise HTTPException(
            status_code=404,
            detail=f"Custom provider '{provider_id}' not found",
        )

    del definitions[provider_id]
    save_custom_providers_to_managed_env(definitions)

    # Trigger provider generation refresh
    await services.admin.refresh_catalog()
    background_tasks.add_task(services.admin.request_restart)

    return {"ok": True, "provider_id": provider_id}


def _filtered_values(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key in FIELD_BY_KEY}


def _local_provider_url(provider_id: str, values: dict[str, str]) -> str:
    if provider_id == "lmstudio":
        return values.get("LM_STUDIO_BASE_URL", "")
    if provider_id == "llamacpp":
        return values.get("LLAMACPP_BASE_URL", "")
    if provider_id == "ollama":
        return values.get("OLLAMA_BASE_URL", "")
    return ""


async def _check_local_provider(
    provider_id: str, base_url: str, path: str
) -> dict[str, Any]:
    clean_url = base_url.strip().rstrip("/")
    if not clean_url:
        return {
            "provider_id": provider_id,
            "status": "missing_url",
            "label": "Missing URL",
            "base_url": base_url,
        }

    url = f"{clean_url}{path}"
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            response = await client.get(url)
        ok = 200 <= response.status_code < 300
        return {
            "provider_id": provider_id,
            "status": "reachable" if ok else "offline",
            "label": "Reachable" if ok else "Offline",
            "base_url": base_url,
            "status_code": response.status_code,
        }
    except Exception as exc:
        return {
            "provider_id": provider_id,
            "status": "offline",
            "label": "Offline",
            "base_url": base_url,
            "error_type": type(exc).__name__,
        }
