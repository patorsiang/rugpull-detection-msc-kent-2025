from __future__ import annotations
from typing import Dict, Any, Iterable, List, Optional

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from backend.utils.constants import DATA_PATH

# Any field name that should become a dropdown of CSV files
FILE_FIELDS: set[str] = {"filename", "source", "target", "eval_source", "new_source"}


def _list_csv_files() -> List[str]:
    return sorted([p.name for p in DATA_PATH.glob("*.csv")])


def _inject_enums_into_components(openapi_schema: Dict[str, Any], files: Iterable[str]) -> None:
    """
    Add enum lists to component schemas (used by body models).
    """
    comps = openapi_schema.get("components", {}).get("schemas", {})
    for _schema_name, schema in comps.items():
        props = schema.get("properties", {})
        if not isinstance(props, dict):
            continue
        for prop_name, spec in props.items():
            if prop_name in FILE_FIELDS and isinstance(spec, dict):
                # Only touch strings to avoid breaking refs/oneOf etc.
                if spec.get("type") == "string":
                    spec["enum"] = list(files)
                    # optional cosmetic name list for some UIs
                    spec["x-enumNames"] = list(files)


def _inject_enums_into_parameters(openapi_schema: Dict[str, Any], files: Iterable[str]) -> None:
    """
    Add enum lists to path/operation parameters (used by Depends() query models).
    """
    for _path, methods in (openapi_schema.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for _method, op in methods.items():
            if not isinstance(op, dict):
                continue
            params = op.get("parameters", [])
            if not isinstance(params, list):
                continue
            for p in params:
                try:
                    name = p.get("name")
                    if name in FILE_FIELDS:
                        schema = p.get("schema") or {}
                        if schema.get("type") == "string":
                            schema["enum"] = list(files)
                            schema["x-enumNames"] = list(files)
                            p["schema"] = schema
                except Exception:
                    # don't let a docs-only failure affect the app
                    continue


def attach_dynamic_file_enums(app: FastAPI) -> None:
    """
    Patch app.openapi() so every time /openapi.json is requested, we:
      1) build a fresh schema,
      2) inject current CSV filenames as enums into components AND parameters,
      3) return the updated schema (Swagger will then render dropdowns).
    """
    def custom_openapi() -> Dict[str, Any]:
        # Always rebuild fresh (avoid stale cache)
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        files = _list_csv_files()
        _inject_enums_into_components(schema, files)
        _inject_enums_into_parameters(schema, files)

        # Do not set app.openapi_schema, so it regenerates each request.
        # If you prefer caching per-process, uncomment the next line:
        # app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[assignment]
