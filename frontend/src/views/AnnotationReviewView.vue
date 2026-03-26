<script setup lang="ts">
import { ref, computed, watch, nextTick, onMounted, onUnmounted } from 'vue'
import { useUiStore } from '@/stores/ui'
import { loadAnnotations, saveAnnotations, getImageUrl } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import {
  FolderOpen,
  Save,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Eye,
  EyeOff,
  Tag,
  Trash2,
  Search,
  ImageIcon,
  AlertCircle,
  CheckCircle2,
} from 'lucide-vue-next'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CocoImage {
  id: number
  file_name: string
  width: number
  height: number
}

interface CocoAnnotation {
  id: number
  image_id: number
  category_id: number
  bbox: [number, number, number, number]
  area: number
  segmentation?: number[][]
}

interface CocoCategory {
  id: number
  name: string
}

interface CocoDataset {
  images: CocoImage[]
  annotations: CocoAnnotation[]
  categories: CocoCategory[]
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CATEGORY_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
  '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#14b8a6',
]

const ZOOM_STEP = 0.2
const ZOOM_MIN = 0.1
const ZOOM_MAX = 5.0

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

const uiStore = useUiStore()

// ---------------------------------------------------------------------------
// Dataset loader state
// ---------------------------------------------------------------------------

const cocoJsonPath = ref('')
const imagesDirPath = ref('')
const isLoadingDataset = ref(false)
const loadError = ref<string | null>(null)

// ---------------------------------------------------------------------------
// Dataset state
// ---------------------------------------------------------------------------

const dataset = ref<CocoDataset | null>(null)
const deletedAnnotationIds = ref<Set<number>>(new Set())
const hasUnsavedChanges = ref(false)
const isSaving = ref(false)

// ---------------------------------------------------------------------------
// Navigator state
// ---------------------------------------------------------------------------

const searchQuery = ref('')
const activeImageId = ref<number | null>(null)

// ---------------------------------------------------------------------------
// Canvas state
// ---------------------------------------------------------------------------

const canvasContainer = ref<HTMLDivElement | null>(null)
const imageEl = ref<HTMLImageElement | null>(null)
const zoom = ref(1.0)
const imageNaturalWidth = ref(0)
const imageNaturalHeight = ref(0)
const imageRenderedWidth = ref(0)
const imageRenderedHeight = ref(0)
const showBboxes = ref(true)
const showLabels = ref(true)
const selectedCategoryFilter = ref<number | null>(null)

// ---------------------------------------------------------------------------
// Computed: filtered image list
// ---------------------------------------------------------------------------

const filteredImages = computed(() => {
  if (!dataset.value) return []
  const query = searchQuery.value.trim().toLowerCase()
  return dataset.value.images.filter(img =>
    !query || img.file_name.toLowerCase().includes(query)
  )
})

const totalImages = computed(() => dataset.value?.images.length ?? 0)

// ---------------------------------------------------------------------------
// Computed: annotation counts per image
// ---------------------------------------------------------------------------

const annotationCountByImage = computed(() => {
  const counts = new Map<number, number>()
  if (!dataset.value) return counts
  for (const ann of dataset.value.annotations) {
    if (!deletedAnnotationIds.value.has(ann.id)) {
      counts.set(ann.image_id, (counts.get(ann.image_id) ?? 0) + 1)
    }
  }
  return counts
})

// ---------------------------------------------------------------------------
// Computed: active image object
// ---------------------------------------------------------------------------

const activeImage = computed(() =>
  dataset.value?.images.find(img => img.id === activeImageId.value) ?? null
)

// ---------------------------------------------------------------------------
// Computed: active image URL
// ---------------------------------------------------------------------------

const activeImageUrl = computed(() => {
  if (!activeImage.value || !imagesDirPath.value) return ''
  const sep = imagesDirPath.value.endsWith('/') ? '' : '/'
  return getImageUrl(imagesDirPath.value + sep + activeImage.value.file_name)
})

// ---------------------------------------------------------------------------
// Computed: active annotations (not deleted)
// ---------------------------------------------------------------------------

const activeAnnotations = computed(() => {
  if (!dataset.value || activeImageId.value === null) return []
  return dataset.value.annotations.filter(ann => {
    if (ann.image_id !== activeImageId.value) return false
    if (deletedAnnotationIds.value.has(ann.id)) return false
    if (selectedCategoryFilter.value !== null && ann.category_id !== selectedCategoryFilter.value) return false
    return true
  })
})

// ---------------------------------------------------------------------------
// Computed: category map for quick lookup
// ---------------------------------------------------------------------------

const categoryMap = computed(() => {
  const map = new Map<number, CocoCategory>()
  dataset.value?.categories.forEach(c => map.set(c.id, c))
  return map
})

// ---------------------------------------------------------------------------
// Computed: category color by category_id (deterministic, cycles through palette)
// ---------------------------------------------------------------------------

function getCategoryColor(categoryId: number): string {
  return CATEGORY_COLORS[(categoryId - 1) % CATEGORY_COLORS.length]
}

function getCategoryName(categoryId: number): string {
  return categoryMap.value.get(categoryId)?.name ?? `cat_${categoryId}`
}

// ---------------------------------------------------------------------------
// Computed: SVG overlay bounding boxes scaled to rendered image size
// ---------------------------------------------------------------------------

const scaledBboxes = computed(() => {
  if (
    !showBboxes.value ||
    imageNaturalWidth.value === 0 ||
    imageRenderedWidth.value === 0
  ) return []

  const scaleX = imageRenderedWidth.value / imageNaturalWidth.value
  const scaleY = imageRenderedHeight.value / imageNaturalHeight.value

  return activeAnnotations.value.map(ann => {
    const [x, y, w, h] = ann.bbox
    return {
      id: ann.id,
      x: x * scaleX,
      y: y * scaleY,
      w: w * scaleX,
      h: h * scaleY,
      color: getCategoryColor(ann.category_id),
      label: getCategoryName(ann.category_id),
      categoryId: ann.category_id,
    }
  })
})

// ---------------------------------------------------------------------------
// Actions: load dataset
// ---------------------------------------------------------------------------

async function handleLoadDataset() {
  if (!cocoJsonPath.value.trim()) {
    loadError.value = 'Please provide a COCO JSON file path.'
    return
  }
  if (!imagesDirPath.value.trim()) {
    loadError.value = 'Please provide the images directory path.'
    return
  }

  isLoadingDataset.value = true
  loadError.value = null

  try {
    const data = await loadAnnotations(cocoJsonPath.value.trim())
    dataset.value = data as CocoDataset
    deletedAnnotationIds.value = new Set()
    hasUnsavedChanges.value = false
    activeImageId.value = dataset.value.images[0]?.id ?? null
    uiStore.showSuccess('Dataset loaded', `${dataset.value.images.length} images, ${dataset.value.annotations.length} annotations`)
  } catch (err: any) {
    loadError.value = err?.response?.data?.detail ?? err?.message ?? 'Failed to load dataset.'
    uiStore.showError('Load failed', loadError.value ?? '')
  } finally {
    isLoadingDataset.value = false
  }
}

// ---------------------------------------------------------------------------
// Actions: select image from navigator
// ---------------------------------------------------------------------------

function selectImage(imageId: number) {
  activeImageId.value = imageId
  zoom.value = 1.0
  imageRenderedWidth.value = 0
  imageRenderedHeight.value = 0
}

// ---------------------------------------------------------------------------
// Actions: image loaded callback — capture natural + rendered dimensions
// ---------------------------------------------------------------------------

function onImageLoad() {
  if (!imageEl.value) return
  imageNaturalWidth.value = imageEl.value.naturalWidth
  imageNaturalHeight.value = imageEl.value.naturalHeight
  updateRenderedSize()
}

function updateRenderedSize() {
  if (!imageEl.value) return
  imageRenderedWidth.value = imageEl.value.offsetWidth
  imageRenderedHeight.value = imageEl.value.offsetHeight
}

// Re-measure when zoom changes
watch(zoom, () => nextTick(updateRenderedSize))

// ---------------------------------------------------------------------------
// Actions: zoom
// ---------------------------------------------------------------------------

function zoomIn() {
  zoom.value = Math.min(ZOOM_MAX, parseFloat((zoom.value + ZOOM_STEP).toFixed(2)))
}

function zoomOut() {
  zoom.value = Math.max(ZOOM_MIN, parseFloat((zoom.value - ZOOM_STEP).toFixed(2)))
}

function fitToView() {
  zoom.value = 1.0
}

// ---------------------------------------------------------------------------
// Actions: delete annotation
// ---------------------------------------------------------------------------

function deleteAnnotation(annotationId: number) {
  deletedAnnotationIds.value.add(annotationId)
  hasUnsavedChanges.value = true
}

// ---------------------------------------------------------------------------
// Actions: save
// ---------------------------------------------------------------------------

async function handleSave() {
  if (!dataset.value || !cocoJsonPath.value) return

  isSaving.value = true
  try {
    const filteredAnnotations = dataset.value.annotations.filter(
      ann => !deletedAnnotationIds.value.has(ann.id)
    )
    const payload = {
      coco_json_path: cocoJsonPath.value.trim(),
      data: {
        images: dataset.value.images,
        annotations: filteredAnnotations,
        categories: dataset.value.categories,
      },
    }
    await saveAnnotations(payload)
    hasUnsavedChanges.value = false
    uiStore.showSuccess('Saved', 'Annotations saved successfully.')
  } catch (err: any) {
    uiStore.showError('Save failed', err?.response?.data?.detail ?? err?.message ?? 'Unknown error')
  } finally {
    isSaving.value = false
  }
}

// ---------------------------------------------------------------------------
// Helpers: format bbox for display
// ---------------------------------------------------------------------------

function formatBbox(bbox: [number, number, number, number]): string {
  return `[${bbox.map(v => Math.round(v)).join(', ')}]`
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

function navigateImage(direction: 'prev' | 'next') {
  if (!dataset.value || filteredImages.value.length === 0) return
  const currentIdx = filteredImages.value.findIndex(img => img.id === activeImageId.value)
  if (direction === 'next' && currentIdx < filteredImages.value.length - 1) {
    selectImage(filteredImages.value[currentIdx + 1].id)
  } else if (direction === 'prev' && currentIdx > 0) {
    selectImage(filteredImages.value[currentIdx - 1].id)
  }
}

function handleKeyDown(e: KeyboardEvent) {
  // Don't trigger shortcuts when typing in inputs
  if ((e.target as HTMLElement)?.tagName === 'INPUT' || (e.target as HTMLElement)?.tagName === 'TEXTAREA') return

  switch (e.key) {
    case 'ArrowLeft':
    case 'ArrowUp':
      e.preventDefault()
      navigateImage('prev')
      break
    case 'ArrowRight':
    case 'ArrowDown':
      e.preventDefault()
      navigateImage('next')
      break
    case '+':
    case '=':
      e.preventDefault()
      zoomIn()
      break
    case '-':
      e.preventDefault()
      zoomOut()
      break
    case 'b':
      showBboxes.value = !showBboxes.value
      break
    case 'l':
      showLabels.value = !showLabels.value
      break
    case 's':
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault()
        if (hasUnsavedChanges.value) handleSave()
      }
      break
  }
}

onMounted(() => window.addEventListener('keydown', handleKeyDown))
onUnmounted(() => window.removeEventListener('keydown', handleKeyDown))
</script>

<template>
  <div class="flex flex-col h-full min-h-0 gap-4" :data-unsaved="hasUnsavedChanges">

    <!-- ------------------------------------------------------------------ -->
    <!-- Header                                                               -->
    <!-- ------------------------------------------------------------------ -->
    <div class="flex items-center justify-between flex-shrink-0">
      <div>
        <h2 class="text-2xl font-bold text-white">Annotation Review</h2>
        <p class="mt-1 text-sm text-gray-400">Review and edit COCO format annotations visually.</p>
      </div>

      <!-- Unsaved indicator -->
      <div v-if="hasUnsavedChanges" class="flex items-center gap-2 text-yellow-400 text-sm">
        <AlertCircle class="h-4 w-4" />
        Unsaved changes
      </div>
      <div v-else-if="dataset" class="flex items-center gap-2 text-green-400 text-sm">
        <CheckCircle2 class="h-4 w-4" />
        Saved
      </div>
    </div>

    <!-- ------------------------------------------------------------------ -->
    <!-- Dataset Loader                                                       -->
    <!-- ------------------------------------------------------------------ -->
    <div class="card p-4 flex-shrink-0">
      <h3 class="text-sm font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <FolderOpen class="h-4 w-4 text-primary" />
        Load Dataset
      </h3>

      <div class="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <!-- COCO JSON path -->
        <DirectoryBrowser
          v-model="cocoJsonPath"
          label="COCO JSON file"
          placeholder="/app/datasets/annotations.json"
          :show-files="true"
          file-pattern="*.json"
          path-mode="input"
          :show-mount-points="true"
          :restrict-to-mounts="false"
        />

        <!-- Images directory -->
        <DirectoryBrowser
          v-model="imagesDirPath"
          label="Images directory"
          placeholder="/app/datasets/images"
          :show-files="false"
          path-mode="input"
          :show-mount-points="true"
          :restrict-to-mounts="false"
        />
      </div>

      <div class="mt-3 flex items-center gap-3">
        <BaseButton
          variant="primary"
          :loading="isLoadingDataset"
          :disabled="isLoadingDataset"
          @click="handleLoadDataset"
        >
          <FolderOpen class="h-4 w-4" />
          Load Dataset
        </BaseButton>

        <span v-if="dataset" class="text-xs text-gray-400">
          {{ totalImages }} images &middot; {{ dataset.categories.length }} categories
        </span>
      </div>

      <AlertBox v-if="loadError" type="error" title="Load error" class="mt-3">
        {{ loadError }}
      </AlertBox>
    </div>

    <!-- ------------------------------------------------------------------ -->
    <!-- Loading overlay for dataset                                          -->
    <!-- ------------------------------------------------------------------ -->
    <div v-if="isLoadingDataset" class="flex justify-center py-12">
      <LoadingSpinner size="lg" message="Loading dataset..." />
    </div>

    <!-- ------------------------------------------------------------------ -->
    <!-- Main workspace (only when dataset is loaded)                         -->
    <!-- ------------------------------------------------------------------ -->
    <div
      v-else-if="dataset"
      class="flex gap-3 min-h-0 flex-1"
      style="min-height: 0; height: calc(100vh - 320px);"
    >

      <!-- ---------------------------------------------------------------- -->
      <!-- Left panel: Image Navigator                                       -->
      <!-- ---------------------------------------------------------------- -->
      <div class="flex flex-col card flex-shrink-0 overflow-hidden" style="width: 250px;">

        <!-- Navigator header -->
        <div class="flex-shrink-0 p-3 border-b border-gray-700">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Images</span>
            <span class="text-xs text-gray-500">{{ filteredImages.length }} / {{ totalImages }}</span>
          </div>
          <!-- Search -->
          <div class="relative">
            <Search class="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-gray-500 pointer-events-none" />
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search..."
              class="w-full pl-8 pr-3 py-1.5 text-xs bg-background-tertiary border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary"
            />
          </div>
        </div>

        <!-- Image list -->
        <div class="flex-1 overflow-y-auto">
          <div
            v-for="img in filteredImages"
            :key="img.id"
            @click="selectImage(img.id)"
            :class="[
              'flex items-start gap-2 px-3 py-2.5 cursor-pointer transition-colors border-b border-gray-700/40',
              img.id === activeImageId
                ? 'bg-primary/20 border-l-2 border-l-primary'
                : 'hover:bg-gray-700/40',
            ]"
          >
            <ImageIcon class="h-4 w-4 text-gray-500 flex-shrink-0 mt-0.5" />
            <div class="min-w-0 flex-1">
              <p class="text-xs text-gray-200 truncate font-medium" :title="img.file_name">
                {{ img.file_name }}
              </p>
              <p class="text-xs text-gray-500 mt-0.5">
                {{ annotationCountByImage.get(img.id) ?? 0 }} annotations
              </p>
            </div>
          </div>

          <!-- Empty search result -->
          <div v-if="filteredImages.length === 0" class="p-4 text-center text-gray-500 text-xs">
            No images match your search.
          </div>
        </div>
      </div>

      <!-- ---------------------------------------------------------------- -->
      <!-- Center: Canvas + Toolbar                                          -->
      <!-- ---------------------------------------------------------------- -->
      <div class="flex flex-col flex-1 min-w-0 min-h-0 gap-2">

        <!-- Toolbar -->
        <div class="card flex-shrink-0 px-3 py-2 flex items-center gap-2 flex-wrap">

          <!-- Save -->
          <BaseButton
            variant="primary"
            size="sm"
            :loading="isSaving"
            :disabled="isSaving || !hasUnsavedChanges"
            @click="handleSave"
          >
            <Save class="h-3.5 w-3.5" />
            Save
          </BaseButton>

          <div class="w-px h-5 bg-gray-700 mx-1" />

          <!-- Zoom controls -->
          <BaseButton variant="ghost" size="sm" @click="zoomOut" :disabled="zoom <= ZOOM_MIN" title="Zoom out">
            <ZoomOut class="h-3.5 w-3.5" />
          </BaseButton>

          <span class="text-xs text-gray-400 tabular-nums w-12 text-center select-none">
            {{ Math.round(zoom * 100) }}%
          </span>

          <BaseButton variant="ghost" size="sm" @click="zoomIn" :disabled="zoom >= ZOOM_MAX" title="Zoom in">
            <ZoomIn class="h-3.5 w-3.5" />
          </BaseButton>

          <BaseButton variant="ghost" size="sm" @click="fitToView" title="Fit to view">
            <Maximize2 class="h-3.5 w-3.5" />
          </BaseButton>

          <div class="w-px h-5 bg-gray-700 mx-1" />

          <!-- Visibility toggles -->
          <button
            @click="showBboxes = !showBboxes"
            :class="[
              'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors',
              showBboxes ? 'bg-primary/20 text-primary' : 'text-gray-400 hover:bg-gray-700/50',
            ]"
            title="Toggle bounding boxes"
          >
            <component :is="showBboxes ? Eye : EyeOff" class="h-3.5 w-3.5" />
            Boxes
          </button>

          <button
            @click="showLabels = !showLabels"
            :class="[
              'flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-colors',
              showLabels ? 'bg-primary/20 text-primary' : 'text-gray-400 hover:bg-gray-700/50',
            ]"
            title="Toggle labels"
          >
            <Tag class="h-3.5 w-3.5" />
            Labels
          </button>

          <!-- Right side: image info -->
          <div class="ml-auto text-xs text-gray-500 select-none">
            <span v-if="activeImage">
              {{ activeImage.file_name }}
              &mdash;
              {{ activeImage.width }} &times; {{ activeImage.height }}px
            </span>
          </div>
        </div>

        <!-- Canvas area -->
        <div
          ref="canvasContainer"
          class="card flex-1 overflow-auto flex items-start justify-start min-h-0 bg-background"
          style="cursor: default;"
        >
          <!-- No image selected -->
          <div
            v-if="!activeImage"
            class="flex-1 flex items-center justify-center h-full text-gray-500"
          >
            <div class="text-center">
              <ImageIcon class="h-16 w-16 mx-auto mb-3 opacity-30" />
              <p class="text-sm">Select an image from the left panel</p>
            </div>
          </div>

          <!-- Image + SVG overlay -->
          <div
            v-else
            class="relative inline-block m-4"
            :style="{ transform: `scale(${zoom})`, transformOrigin: 'top left' }"
          >
            <img
              ref="imageEl"
              :src="activeImageUrl"
              :alt="activeImage.file_name"
              class="block max-w-none"
              :style="{ display: 'block' }"
              @load="onImageLoad"
              @error="() => uiStore.showError('Image error', 'Could not load image.')"
            />

            <!-- SVG overlay for bounding boxes -->
            <svg
              v-if="imageRenderedWidth > 0"
              :width="imageRenderedWidth"
              :height="imageRenderedHeight"
              class="absolute top-0 left-0 pointer-events-none overflow-visible"
              :viewBox="`0 0 ${imageRenderedWidth} ${imageRenderedHeight}`"
            >
              <g v-for="bbox in scaledBboxes" :key="bbox.id">
                <!-- Rectangle -->
                <rect
                  :x="bbox.x"
                  :y="bbox.y"
                  :width="bbox.w"
                  :height="bbox.h"
                  :stroke="bbox.color"
                  stroke-width="2"
                  fill="none"
                  opacity="0.9"
                />

                <!-- Label background + text -->
                <template v-if="showLabels">
                  <rect
                    :x="bbox.x"
                    :y="bbox.y - 18"
                    :width="bbox.label.length * 7 + 10"
                    :height="18"
                    :fill="bbox.color"
                    rx="3"
                    ry="3"
                    opacity="0.9"
                  />
                  <text
                    :x="bbox.x + 5"
                    :y="bbox.y - 5"
                    fill="white"
                    font-size="11"
                    font-family="ui-monospace, monospace"
                    font-weight="600"
                  >{{ bbox.label }}</text>
                </template>
              </g>
            </svg>
          </div>
        </div>
      </div>

      <!-- ---------------------------------------------------------------- -->
      <!-- Right panel: Annotation list                                      -->
      <!-- ---------------------------------------------------------------- -->
      <div class="flex flex-col card flex-shrink-0 overflow-hidden" style="width: 300px;">

        <!-- Panel header -->
        <div class="flex-shrink-0 px-3 py-3 border-b border-gray-700">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-semibold text-gray-400 uppercase tracking-wider">Annotations</span>
            <span class="text-xs font-medium text-gray-400 bg-background-tertiary px-2 py-0.5 rounded-full">
              {{ activeAnnotations.length }}
            </span>
          </div>
          <!-- Category filter -->
          <select
            v-model="selectedCategoryFilter"
            class="w-full text-xs bg-background-tertiary border border-gray-600 rounded-lg px-2 py-1.5 text-gray-300 focus:outline-none focus:border-primary"
          >
            <option :value="null">All categories</option>
            <option v-for="cat in dataset.categories" :key="cat.id" :value="cat.id">
              {{ cat.name }}
            </option>
          </select>
        </div>

        <!-- Annotation list -->
        <div class="flex-1 overflow-y-auto">

          <!-- Empty state -->
          <div
            v-if="activeAnnotations.length === 0"
            class="flex flex-col items-center justify-center h-32 text-gray-600"
          >
            <Tag class="h-8 w-8 mb-2 opacity-40" />
            <p class="text-xs">No annotations for this image</p>
          </div>

          <!-- Annotation items -->
          <div
            v-for="ann in activeAnnotations"
            :key="ann.id"
            class="flex items-start gap-2 px-3 py-2.5 border-b border-gray-700/40 hover:bg-gray-700/20 group transition-colors"
          >
            <!-- Color dot -->
            <div
              class="flex-shrink-0 w-2.5 h-2.5 rounded-full mt-1"
              :style="{ backgroundColor: getCategoryColor(ann.category_id) }"
            />

            <!-- Annotation details -->
            <div class="flex-1 min-w-0">
              <p class="text-xs font-semibold text-gray-200 truncate">
                {{ getCategoryName(ann.category_id) }}
              </p>
              <p class="text-xs text-gray-500 mt-0.5 font-mono">
                {{ formatBbox(ann.bbox) }}
              </p>
              <p class="text-xs text-gray-600 mt-0.5">
                area: {{ Math.round(ann.area) }}
              </p>
            </div>

            <!-- Delete button -->
            <button
              @click="deleteAnnotation(ann.id)"
              class="flex-shrink-0 p-1 rounded-md text-gray-600 hover:text-red-400 hover:bg-red-400/10 transition-colors opacity-0 group-hover:opacity-100"
              title="Delete annotation"
            >
              <Trash2 class="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <!-- Category legend -->
        <div class="flex-shrink-0 border-t border-gray-700 p-3">
          <p class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Categories</p>
          <div class="space-y-1 max-h-36 overflow-y-auto">
            <div
              v-for="cat in dataset.categories"
              :key="cat.id"
              class="flex items-center gap-2"
            >
              <div
                class="w-2.5 h-2.5 rounded-sm flex-shrink-0"
                :style="{ backgroundColor: getCategoryColor(cat.id) }"
              />
              <span class="text-xs text-gray-400 truncate">{{ cat.name }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ------------------------------------------------------------------ -->
    <!-- Empty state (no dataset loaded, not loading)                         -->
    <!-- ------------------------------------------------------------------ -->
    <div v-else-if="!isLoadingDataset" class="card p-12 text-center text-gray-500 flex-shrink-0">
      <FolderOpen class="h-16 w-16 mx-auto mb-4 opacity-20" />
      <p class="text-sm font-medium">No dataset loaded</p>
      <p class="text-xs mt-1">Fill in the paths above and click "Load Dataset" to begin.</p>
    </div>

  </div>
</template>
