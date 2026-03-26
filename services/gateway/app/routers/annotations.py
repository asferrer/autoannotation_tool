"""
Annotations Router - COCO annotation CRUD operations.
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Annotations"], prefix="/annotations")

SEGMENTATION_SERVICE_URL = os.environ.get(
    "SEGMENTATION_SERVICE_URL", "http://segmentation:8002"
)


# ---------- Schemas ----------

class LoadAnnotationsRequest(BaseModel):
    coco_json_path: str


class SaveAnnotationsRequest(BaseModel):
    coco_json_path: str
    data: Dict[str, Any]


class CreateAnnotationRequest(BaseModel):
    coco_json_path: str
    image_id: int
    category_id: int
    bbox: List[float]
    segmentation: Optional[List[List[float]]] = None
    area: Optional[float] = None


class UpdateAnnotationRequest(BaseModel):
    coco_json_path: str
    annotation_id: int
    bbox: Optional[List[float]] = None
    category_id: Optional[int] = None
    segmentation: Optional[List[List[float]]] = None


class DeleteAnnotationRequest(BaseModel):
    coco_json_path: str
    annotation_id: int


class ReannotateRequest(BaseModel):
    coco_json_path: str
    image_id: int
    annotation_id: int


# ---------- Helpers ----------

def _read_coco_sync(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise HTTPException(404, f"File not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_coco_sync(path: str, data: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        backup = p.with_suffix(
            f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(p, backup)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


async def _read_coco(path: str) -> Dict[str, Any]:
    return await asyncio.to_thread(_read_coco_sync, path)


async def _write_coco(path: str, data: Dict[str, Any]):
    await asyncio.to_thread(_write_coco_sync, path, data)


# ---------- Endpoints ----------

@router.post("/load")
async def load_annotations(request: LoadAnnotationsRequest):
    """Load a COCO JSON file and return its contents."""
    data = await _read_coco(request.coco_json_path)
    return {
        "images": data.get("images", []),
        "annotations": data.get("annotations", []),
        "categories": data.get("categories", []),
        "info": data.get("info", {}),
    }


@router.get("/images")
async def list_images(coco_json_path: str = Query(...)):
    """List all images in the dataset with annotation counts."""
    data = await _read_coco(coco_json_path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])

    ann_counts: Dict[int, int] = {}
    for ann in annotations:
        img_id = ann.get("image_id")
        ann_counts[img_id] = ann_counts.get(img_id, 0) + 1

    return [
        {**img, "annotation_count": ann_counts.get(img["id"], 0)}
        for img in images
    ]


@router.get("/images/{image_id}")
async def get_image_annotations(image_id: int, coco_json_path: str = Query(...)):
    """Get all annotations for a specific image."""
    data = await _read_coco(coco_json_path)
    annotations = [
        a for a in data.get("annotations", []) if a.get("image_id") == image_id
    ]
    image = next((i for i in data.get("images", []) if i["id"] == image_id), None)
    return {
        "image": image,
        "annotations": annotations,
        "categories": data.get("categories", []),
    }


@router.post("/save")
async def save_annotations(request: SaveAnnotationsRequest):
    """Save complete COCO JSON data."""
    await _write_coco(request.coco_json_path, request.data)
    return {"status": "saved", "path": request.coco_json_path}


@router.post("/create")
async def create_annotation(request: CreateAnnotationRequest):
    """Create a new annotation."""
    data = await _read_coco(request.coco_json_path)
    annotations = data.get("annotations", [])

    max_id = max((a["id"] for a in annotations), default=0)
    new_ann: Dict[str, Any] = {
        "id": max_id + 1,
        "image_id": request.image_id,
        "category_id": request.category_id,
        "bbox": request.bbox,
        "area": request.area or (request.bbox[2] * request.bbox[3]),
        "iscrowd": 0,
    }
    if request.segmentation:
        new_ann["segmentation"] = request.segmentation

    data["annotations"].append(new_ann)
    await _write_coco(request.coco_json_path, data)
    return new_ann


@router.put("/update")
async def update_annotation(request: UpdateAnnotationRequest):
    """Update an existing annotation."""
    data = await _read_coco(request.coco_json_path)
    ann = next(
        (a for a in data.get("annotations", []) if a["id"] == request.annotation_id),
        None,
    )
    if not ann:
        raise HTTPException(404, f"Annotation {request.annotation_id} not found")

    if request.bbox is not None:
        ann["bbox"] = request.bbox
        ann["area"] = request.bbox[2] * request.bbox[3]
    if request.category_id is not None:
        ann["category_id"] = request.category_id
    if request.segmentation is not None:
        ann["segmentation"] = request.segmentation

    await _write_coco(request.coco_json_path, data)
    return ann


@router.post("/delete")
async def delete_annotation(request: DeleteAnnotationRequest):
    """Delete an annotation."""
    data = await _read_coco(request.coco_json_path)
    data["annotations"] = [
        a for a in data.get("annotations", []) if a["id"] != request.annotation_id
    ]
    await _write_coco(request.coco_json_path, data)
    return {"status": "deleted", "annotation_id": request.annotation_id}


@router.post("/reannotate")
async def reannotate_with_sam3(request: ReannotateRequest):
    """Re-annotate an image region using SAM3."""
    data = await _read_coco(request.coco_json_path)
    ann = next(
        (a for a in data.get("annotations", []) if a["id"] == request.annotation_id),
        None,
    )
    if not ann:
        raise HTTPException(404, f"Annotation {request.annotation_id} not found")

    image = next(
        (i for i in data.get("images", []) if i["id"] == request.image_id), None
    )
    if not image:
        raise HTTPException(404, f"Image {request.image_id} not found")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{SEGMENTATION_SERVICE_URL}/sam3/segment-image",
                json={
                    "image_path": image.get("file_name", ""),
                    "bbox": ann.get("bbox"),
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                if result.get("segmentation"):
                    ann["segmentation"] = result["segmentation"]
                    await _write_coco(request.coco_json_path, data)
                return {"status": "reannotated", "annotation": ann}
            raise HTTPException(resp.status_code, "SAM3 reannotation failed")
    except httpx.RequestError as e:
        raise HTTPException(503, f"Segmentation service unavailable: {e}")
