"""
Filesystem Router - Browse directories and serve images.
Validates all paths against allowed base directories to prevent traversal attacks.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
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
    if not any(p == base or p.is_relative_to(base) for base in ALLOWED_BASE_PATHS):
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


@router.get("/scan-images")
async def scan_images(path: str = Query(...)):
    """Scan a directory recursively for images and return COCO-compatible image list."""
    import asyncio as _asyncio

    p = _validate_path(path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"Directory not found: {path}")

    def _scan_sync() -> list:
        from PIL import Image as PILImage
        results = []
        img_id = 1
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    with PILImage.open(f) as img:
                        w, h = img.size
                    rel_path = f.relative_to(p)
                    results.append({
                        "id": img_id,
                        "file_name": str(rel_path).replace("\\", "/"),
                        "width": w,
                        "height": h,
                    })
                    img_id += 1
                except Exception:
                    continue
        return results

    images = await _asyncio.to_thread(_scan_sync)
    return {"images": images, "directory": str(p)}


UPLOAD_BASE = Path(os.environ.get("UPLOAD_BASE", "/app/datasets")).resolve()


@router.post("/upload")
async def upload_files(
    task_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Upload image files from the user's machine into a task directory."""
    # Sanitize task name
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in task_name).strip() or "upload"
    dest_dir = UPLOAD_BASE / safe_name / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    coco_path = ""
    for f in files:
        if not f.filename:
            continue

        # Handle COCO JSON files
        if f.filename.endswith(".json"):
            dest_json = UPLOAD_BASE / safe_name / Path(f.filename).name
            dest_json.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_json, "wb") as out:
                shutil.copyfileobj(f.file, out)
            coco_path = str(dest_json)
            continue

        ext = Path(f.filename).suffix.lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        # Preserve subdirectory structure from folder uploads
        # Browser sends paths like "subfolder/image.jpg"
        # Strip path traversal components before constructing the destination
        raw_parts = Path(f.filename.replace("\\", "/")).parts
        safe_parts = [p for p in raw_parts if p not in ("..", ".") and p]
        if not safe_parts:
            continue
        safe_rel = Path(*safe_parts)
        dest_path = (dest_dir / safe_rel).resolve()
        # Final guard: ensure the resolved path stays inside dest_dir
        if not dest_path.is_relative_to(dest_dir.resolve()):
            logger.warning(f"Upload rejected (path traversal attempt): {f.filename!r}")
            continue
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        uploaded.append(str(safe_rel).replace("\\", "/"))

    return {
        "directory": str(dest_dir),
        "uploaded_count": len(uploaded),
        "files": uploaded,
        "coco_json_path": coco_path,
    }


@router.post("/upload-coco")
async def upload_coco(
    task_name: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a COCO JSON file."""
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in task_name).strip() or "upload"
    dest_dir = UPLOAD_BASE / safe_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(file.filename or "annotations.json").name
    dest_path = dest_dir / filename
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    return {"path": str(dest_path)}


@router.get("/image")
async def serve_image(path: str = Query(...)):
    """Serve an image file by absolute path.

    If the exact path doesn't exist, tries to find the file by name
    in parent directories (handles COCO relative paths).
    """
    p = _validate_path(path)

    if not p.exists() or not p.is_file():
        # Try searching: maybe the path has extra or missing subdirectories
        # Search by filename in the nearest valid ancestor
        filename = p.name
        search_dir = p.parent
        while not search_dir.exists() and len(search_dir.parts) > 2:
            search_dir = search_dir.parent

        if search_dir.exists():
            matches = list(search_dir.rglob(filename))
            if matches:
                p = matches[0]

    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"Image not found: {path}")

    media_type = MEDIA_TYPES.get(p.suffix.lower(), "application/octet-stream")
    return FileResponse(str(p), media_type=media_type)
