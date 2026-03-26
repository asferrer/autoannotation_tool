"""
Datasets Router - Dataset listing, analysis, and category management.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Datasets"], prefix="/datasets")

DATASETS_BASE = Path("/app/datasets")


# ---------- Schemas ----------

class AnalyzeRequest(BaseModel):
    dataset_path: str


class RenameCategoryRequest(BaseModel):
    dataset_path: str
    category_id: int
    new_name: str


# ---------- Helpers ----------

def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------- Endpoints ----------

@router.get("")
async def list_datasets():
    """List all COCO JSON datasets found in the datasets directory."""
    datasets: List[dict] = []

    if not DATASETS_BASE.exists():
        return datasets

    for json_file in DATASETS_BASE.rglob("*.json"):
        try:
            data = _load_json(json_file)
            if "images" in data and "annotations" in data:
                datasets.append({
                    "name": json_file.stem,
                    "path": str(json_file),
                    "num_images": len(data.get("images", [])),
                    "num_annotations": len(data.get("annotations", [])),
                    "num_categories": len(data.get("categories", [])),
                })
        except (json.JSONDecodeError, IOError):
            continue

    return datasets


@router.post("/analyze")
async def analyze_dataset(request: AnalyzeRequest):
    """Analyze a COCO JSON dataset and return statistics."""
    p = Path(request.dataset_path)
    if not p.exists():
        raise HTTPException(404, f"Dataset not found: {request.dataset_path}")

    data = _load_json(p)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    cat_counts: Dict[int, int] = {}
    for ann in annotations:
        cat_id = ann.get("category_id")
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1

    cat_info = [
        {"id": cat["id"], "name": cat["name"], "count": cat_counts.get(cat["id"], 0)}
        for cat in categories
    ]

    img_id_to_idx = {img["id"]: i for i, img in enumerate(images)}
    ann_per_image = [0] * len(images)
    for ann in annotations:
        idx = img_id_to_idx.get(ann.get("image_id"))
        if idx is not None:
            ann_per_image[idx] += 1

    mean_ann = sum(ann_per_image) / max(len(ann_per_image), 1)

    return {
        "total_images": len(images),
        "total_annotations": len(annotations),
        "categories": cat_info,
        "annotations_per_image": {
            "mean": mean_ann,
            "min": min(ann_per_image) if ann_per_image else 0,
            "max": max(ann_per_image) if ann_per_image else 0,
        },
    }


@router.post("/categories/rename")
async def rename_category(request: RenameCategoryRequest):
    """Rename a category in the dataset."""
    p = Path(request.dataset_path)
    if not p.exists():
        raise HTTPException(404, "Dataset not found")

    data = _load_json(p)
    cat = next(
        (c for c in data.get("categories", []) if c["id"] == request.category_id),
        None,
    )
    if not cat:
        raise HTTPException(404, f"Category {request.category_id} not found")

    old_name = cat["name"]
    cat["name"] = request.new_name
    _save_json(p, data)

    return {"status": "renamed", "old_name": old_name, "new_name": request.new_name}


@router.delete("/categories/{category_id}")
async def delete_category(category_id: int, dataset_path: str = Query(...)):
    """Delete a category and all its annotations from the dataset."""
    p = Path(dataset_path)
    if not p.exists():
        raise HTTPException(404, "Dataset not found")

    data = _load_json(p)
    data["categories"] = [
        c for c in data.get("categories", []) if c["id"] != category_id
    ]
    removed = sum(
        1 for a in data.get("annotations", []) if a.get("category_id") == category_id
    )
    data["annotations"] = [
        a for a in data.get("annotations", []) if a.get("category_id") != category_id
    ]
    _save_json(p, data)

    return {
        "status": "deleted",
        "category_id": category_id,
        "annotations_removed": removed,
    }
