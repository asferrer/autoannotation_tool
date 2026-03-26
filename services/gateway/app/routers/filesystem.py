"""
Filesystem Router - Browse directories and serve images.
Validates all paths against allowed base directories to prevent traversal attacks.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Filesystem"], prefix="/filesystem")

ALLOWED_BASE_PATHS = [
    Path(p.strip()).resolve()
    for p in os.environ.get("ALLOWED_FS_PATHS", "/app/datasets").split(",")
]

MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}


def _validate_path(path: str) -> Path:
    """Resolve path and ensure it's within allowed base directories."""
    p = Path(path).resolve()
    if not any(p == base or str(p).startswith(str(base) + os.sep) for base in ALLOWED_BASE_PATHS):
        raise HTTPException(403, "Access denied: path outside allowed directories")
    return p


@router.get("/browse")
async def browse(
    path: str = Query("/app/datasets"),
    type: str = Query("directories"),
    pattern: Optional[str] = None,
) -> List[str]:
    """Browse filesystem directories or files at the given path."""
    p = _validate_path(path)
    if not p.exists():
        raise HTTPException(404, f"Path not found: {path}")

    if type == "directories":
        return sorted(str(d) for d in p.iterdir() if d.is_dir())

    if pattern:
        return sorted(str(f) for f in p.glob(pattern) if f.is_file())
    return sorted(str(f) for f in p.iterdir() if f.is_file())


@router.get("/check-path")
async def check_path(path: str = Query(...)):
    """Check if a path exists and whether it's a directory."""
    try:
        p = _validate_path(path)
        return {"exists": p.exists(), "is_directory": p.is_dir() if p.exists() else False}
    except HTTPException:
        return {"exists": False, "is_directory": False}


@router.get("/image")
async def serve_image(path: str = Query(...)):
    """Serve an image file by absolute path."""
    p = _validate_path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"Image not found: {path}")

    media_type = MEDIA_TYPES.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media_type)
