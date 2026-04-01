import { ref, computed, shallowRef } from 'vue'

export interface BBox {
  x: number
  y: number
  w: number
  h: number
}

export interface Annotation {
  id: number
  image_id: number
  category_id: number
  bbox: [number, number, number, number]
  area: number
  segmentation?: number[][]
  iscrowd?: number
}

export interface Category {
  id: number
  name: string
}

export interface CocoDataset {
  images: { id: number; file_name: string; width: number; height: number }[]
  annotations: Annotation[]
  categories: Category[]
}

type ActionType = 'create' | 'delete' | 'move' | 'resize' | 'update-category' | 'update-mask'

interface HistoryEntry {
  type: ActionType
  annotation: Annotation
  previous?: Annotation
}

export type ToolMode = 'select' | 'draw-bbox' | 'pan' | 'segment-point'

export function useAnnotationEditor() {
  const dataset = shallowRef<CocoDataset | null>(null)
  const annotations = ref<Annotation[]>([])
  const selectedId = ref<number | null>(null)
  const toolMode = ref<ToolMode>('select')

  const undoStack = ref<HistoryEntry[]>([])
  const redoStack = ref<HistoryEntry[]>([])
  const hasUnsavedChanges = ref(false)
  // JSON snapshot of annotations at last save — used to detect divergence after undo/redo
  let savedSnapshot = ''

  function loadDataset(data: CocoDataset) {
    dataset.value = data
    annotations.value = [...data.annotations]
    selectedId.value = null
    undoStack.value = []
    redoStack.value = []
    hasUnsavedChanges.value = false
    savedSnapshot = JSON.stringify(data.annotations)
  }

  function getAnnotationsForImage(imageId: number): Annotation[] {
    return annotations.value.filter((a) => a.image_id === imageId)
  }

  const selectedAnnotation = computed(() =>
    selectedId.value !== null
      ? annotations.value.find((a) => a.id === selectedId.value) ?? null
      : null
  )

  function select(id: number | null) {
    selectedId.value = id
  }

  // --- History helpers ---
  function pushHistory(entry: HistoryEntry) {
    undoStack.value.push(entry)
    redoStack.value = []
    hasUnsavedChanges.value = true
  }

  // --- CRUD ---
  function nextId(): number {
    const maxId = annotations.value.reduce((max, a) => Math.max(max, a.id), 0)
    return maxId + 1
  }

  function createAnnotation(imageId: number, categoryId: number, bbox: BBox): Annotation {
    const ann: Annotation = {
      id: nextId(),
      image_id: imageId,
      category_id: categoryId,
      bbox: [bbox.x, bbox.y, bbox.w, bbox.h],
      area: bbox.w * bbox.h,
      iscrowd: 0,
    }
    annotations.value = [...annotations.value, ann]
    pushHistory({ type: 'create', annotation: { ...ann } })
    selectedId.value = ann.id
    return ann
  }

  function deleteAnnotation(id: number) {
    const ann = annotations.value.find((a) => a.id === id)
    if (!ann) return
    pushHistory({ type: 'delete', annotation: { ...ann } })
    annotations.value = annotations.value.filter((a) => a.id !== id)
    if (selectedId.value === id) selectedId.value = null
  }

  function moveAnnotation(id: number, dx: number, dy: number) {
    const ann = annotations.value.find((a) => a.id === id)
    if (!ann) return
    const previous = { ...ann, bbox: [...ann.bbox] as [number, number, number, number] }
    ann.bbox[0] += dx
    ann.bbox[1] += dy
    annotations.value = [...annotations.value]
    pushHistory({ type: 'move', annotation: { ...ann, bbox: [...ann.bbox] as [number, number, number, number] }, previous })
  }

  function resizeAnnotation(id: number, bbox: BBox) {
    const ann = annotations.value.find((a) => a.id === id)
    if (!ann) return
    const previous = { ...ann, bbox: [...ann.bbox] as [number, number, number, number] }
    ann.bbox = [bbox.x, bbox.y, bbox.w, bbox.h]
    ann.area = bbox.w * bbox.h
    annotations.value = [...annotations.value]
    pushHistory({ type: 'resize', annotation: { ...ann, bbox: [...ann.bbox] as [number, number, number, number] }, previous })
  }

  function updateCategory(id: number, categoryId: number) {
    const ann = annotations.value.find((a) => a.id === id)
    if (!ann) return
    const previous = { ...ann }
    ann.category_id = categoryId
    annotations.value = [...annotations.value]
    pushHistory({ type: 'update-category', annotation: { ...ann }, previous })
  }

  // Attach a SAM3-generated segmentation to an existing annotation (undo-able).
  function updateAnnotationSegmentation(id: number, segmentation: number[][]) {
    const ann = annotations.value.find((a) => a.id === id)
    if (!ann) return
    const previous = { ...ann, bbox: [...ann.bbox] as [number, number, number, number], segmentation: ann.segmentation?.map((p) => [...p]) }
    ann.segmentation = segmentation
    annotations.value = [...annotations.value]
    pushHistory({ type: 'update-mask', annotation: { ...ann, bbox: [...ann.bbox] as [number, number, number, number] }, previous })
  }

  // Update mask (polygon edit) + auto-update bbox from polygon extents. Undo-able.
  function updateAnnotationMask(id: number, segmentation: number[][], bbox: [number, number, number, number]) {
    const ann = annotations.value.find((a) => a.id === id)
    if (!ann) return
    const previous = { ...ann, bbox: [...ann.bbox] as [number, number, number, number], segmentation: ann.segmentation?.map((p) => [...p]) }
    ann.segmentation = segmentation
    ann.bbox = bbox
    ann.area = bbox[2] * bbox[3]
    annotations.value = [...annotations.value]
    pushHistory({ type: 'update-mask', annotation: { ...ann, bbox: [...ann.bbox] as [number, number, number, number] }, previous })
  }

  // --- Undo / Redo ---
  function undo() {
    const entry = undoStack.value.pop()
    if (!entry) return

    switch (entry.type) {
      case 'create':
        annotations.value = annotations.value.filter((a) => a.id !== entry.annotation.id)
        if (selectedId.value === entry.annotation.id) selectedId.value = null
        break
      case 'delete':
        annotations.value = [...annotations.value, { ...entry.annotation }]
        break
      case 'move':
      case 'resize':
      case 'update-category':
      case 'update-mask':
        if (entry.previous) {
          const idx = annotations.value.findIndex((a) => a.id === entry.annotation.id)
          if (idx >= 0) {
            annotations.value[idx] = { ...entry.previous }
            annotations.value = [...annotations.value]
          }
        }
        break
    }

    redoStack.value.push(entry)
    hasUnsavedChanges.value = JSON.stringify(annotations.value) !== savedSnapshot
  }

  function redo() {
    const entry = redoStack.value.pop()
    if (!entry) return

    switch (entry.type) {
      case 'create':
        annotations.value = [...annotations.value, { ...entry.annotation }]
        break
      case 'delete':
        annotations.value = annotations.value.filter((a) => a.id !== entry.annotation.id)
        if (selectedId.value === entry.annotation.id) selectedId.value = null
        break
      case 'move':
      case 'resize':
      case 'update-category':
      case 'update-mask':
        {
          const idx = annotations.value.findIndex((a) => a.id === entry.annotation.id)
          if (idx >= 0) {
            annotations.value[idx] = { ...entry.annotation }
            annotations.value = [...annotations.value]
          }
        }
        break
    }

    undoStack.value.push(entry)
    hasUnsavedChanges.value = JSON.stringify(annotations.value) !== savedSnapshot
  }

  const canUndo = computed(() => undoStack.value.length > 0)
  const canRedo = computed(() => redoStack.value.length > 0)

  // --- Export for save ---
  function getDatasetForSave(): CocoDataset | null {
    if (!dataset.value) return null
    return {
      images: dataset.value.images,
      annotations: [...annotations.value],
      categories: dataset.value.categories,
    }
  }

  function markSaved() {
    hasUnsavedChanges.value = false
    savedSnapshot = JSON.stringify(annotations.value)
  }

  return {
    dataset,
    annotations,
    selectedId,
    selectedAnnotation,
    toolMode,
    hasUnsavedChanges,
    canUndo,
    canRedo,
    loadDataset,
    getAnnotationsForImage,
    select,
    createAnnotation,
    deleteAnnotation,
    moveAnnotation,
    resizeAnnotation,
    updateCategory,
    updateAnnotationSegmentation,
    updateAnnotationMask,
    undo,
    redo,
    getDatasetForSave,
    markSaved,
  }
}
