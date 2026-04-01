<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue'
import { getPartialAnnotations, getImageUrl } from '@/lib/api'
import type { PartialAnnotationsResponse, CocoAnnotation } from '@/types/api'
import { Eye, Layers } from 'lucide-vue-next'

const props = defineProps<{
  jobId: string
  isRunning: boolean
}>()

const partialData = ref<PartialAnnotationsResponse | null>(null)
const selectedImageId = ref<number | null>(null)
const expanded = ref(false)
const imgRenderedW = ref(0)
const imgRenderedH = ref(0)
let pollTimer: ReturnType<typeof setInterval> | null = null

const CATEGORY_COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#f97316', '#14b8a6', '#a855f7',
]

const images = computed(() => partialData.value?.data?.images ?? [])
const annotations = computed(() => partialData.value?.data?.annotations ?? [])
const categories = computed(() => partialData.value?.data?.categories ?? [])

const selectedImage = computed(() =>
  images.value.find(img => img.id === selectedImageId.value) ?? null
)

const selectedAnnotations = computed(() =>
  annotations.value.filter(ann => ann.image_id === selectedImageId.value)
)

function categoryColor(catId: number): string {
  return CATEGORY_COLORS[(catId - 1) % CATEGORY_COLORS.length]
}

function categoryName(catId: number): string {
  return categories.value.find(c => c.id === catId)?.name ?? `cat-${catId}`
}

function annotationCountForImage(imageId: number): number {
  return annotations.value.filter(a => a.image_id === imageId).length
}

/** Convert COCO flat polygon [x1,y1,x2,y2,...] to SVG points "x1,y1 x2,y2 ..." */
function segmentationToPoints(seg: number[]): string {
  const points: string[] = []
  for (let i = 0; i < seg.length - 1; i += 2) {
    points.push(`${seg[i]},${seg[i + 1]}`)
  }
  return points.join(' ')
}

function hasSegmentation(ann: CocoAnnotation): boolean {
  return !!ann.segmentation && ann.segmentation.length > 0
}

function onImageLoad(e: Event) {
  const img = e.target as HTMLImageElement
  imgRenderedW.value = img.clientWidth
  imgRenderedH.value = img.clientHeight
}

async function fetchPartial() {
  if (!props.jobId) return
  try {
    partialData.value = await getPartialAnnotations(props.jobId)
    const imgs = images.value
    if (imgs.length === 0) return

    if (selectedImageId.value === null) {
      // First load: select the most recent image
      selectedImageId.value = imgs[imgs.length - 1].id
    } else if (!imgs.some(img => img.id === selectedImageId.value)) {
      // Selected image slid out of the preview window → move to the oldest
      // still visible so the user can continue reviewing without losing context
      selectedImageId.value = imgs[0].id
    }
    // If the selected image is still in the list, keep it unchanged
  } catch {
    // ignore polling errors
  }
}

function startPolling() {
  stopPolling()
  fetchPartial()
  pollTimer = setInterval(fetchPartial, 8000)
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

watch(selectedImageId, () => {
  imgRenderedW.value = 0
  imgRenderedH.value = 0
})

watch(() => props.isRunning, (running) => {
  if (running) startPolling()
  else {
    stopPolling()
    fetchPartial()
  }
}, { immediate: true })

onUnmounted(stopPolling)
</script>

<template>
  <div v-if="partialData?.available" class="mt-4">
    <button
      @click="expanded = !expanded"
      class="flex items-center gap-2 text-sm font-medium text-gray-400 hover:text-white transition-colors mb-2"
    >
      <Eye class="h-4 w-4" />
      Live Annotation Preview
      <span class="text-xs text-gray-500">
        ({{ images.length }} images, {{ annotations.length }} annotations)
      </span>
      <span class="text-xs">{{ expanded ? '&#9650;' : '&#9660;' }}</span>
    </button>

    <div v-if="expanded" class="bg-background-tertiary rounded-lg p-3">
      <!-- Vertical stack: image list on top, viewer below -->
      <div class="flex flex-col gap-3">
        <!-- Image list (horizontal scrollable strip) -->
        <div class="flex gap-1.5 overflow-x-auto pb-1.5 min-h-[40px]">
          <button
            v-for="img in images"
            :key="img.id"
            @click="selectedImageId = img.id"
            :class="[
              'flex-shrink-0 px-2.5 py-1.5 rounded text-xs transition-colors whitespace-nowrap',
              selectedImageId === img.id
                ? 'bg-primary/20 text-primary ring-1 ring-primary/40'
                : 'bg-background-secondary text-gray-400 hover:text-gray-200'
            ]"
          >
            {{ img.file_name }}
            <span class="text-[10px] text-gray-500 ml-1">
              ({{ annotationCountForImage(img.id) }})
            </span>
          </button>
        </div>

        <!-- Image viewer with annotation overlays -->
        <div class="relative bg-background-secondary rounded overflow-hidden" style="min-height: 280px; height: 50vh; max-height: 500px;">
          <template v-if="selectedImage">
            <div class="relative w-full h-full flex items-center justify-center">
              <img
                :src="getImageUrl((selectedImage as any).full_path || selectedImage.file_name)"
                :alt="selectedImage.file_name"
                class="max-w-full max-h-full object-contain"
                @load="onImageLoad"
              />
              <!-- SVG overlay: masks + bboxes -->
              <svg
                v-if="imgRenderedW > 0"
                class="absolute pointer-events-none"
                :style="{ width: imgRenderedW + 'px', height: imgRenderedH + 'px' }"
                :viewBox="`0 0 ${selectedImage.width} ${selectedImage.height}`"
              >
                <g v-for="ann in selectedAnnotations" :key="ann.id">
                  <!-- Segmentation mask (filled polygon) -->
                  <template v-if="hasSegmentation(ann)">
                    <polygon
                      v-for="(seg, si) in ann.segmentation"
                      :key="`seg-${ann.id}-${si}`"
                      :points="segmentationToPoints(seg)"
                      :fill="categoryColor(ann.category_id)"
                      fill-opacity="0.25"
                      :stroke="categoryColor(ann.category_id)"
                      stroke-width="1.5"
                    />
                  </template>
                  <!-- Bounding box -->
                  <rect
                    :x="ann.bbox[0]"
                    :y="ann.bbox[1]"
                    :width="ann.bbox[2]"
                    :height="ann.bbox[3]"
                    :stroke="categoryColor(ann.category_id)"
                    stroke-width="2"
                    fill="none"
                    stroke-dasharray="4 2"
                  />
                  <!-- Label -->
                  <rect
                    :x="ann.bbox[0]"
                    :y="Math.max(ann.bbox[1] - 18, 0)"
                    :width="categoryName(ann.category_id).length * 7.5 + 8"
                    height="16"
                    :fill="categoryColor(ann.category_id)"
                    rx="2"
                  />
                  <text
                    :x="ann.bbox[0] + 4"
                    :y="Math.max(ann.bbox[1] - 5, 11)"
                    fill="white"
                    font-size="11"
                    font-weight="600"
                  >
                    {{ categoryName(ann.category_id) }}
                  </text>
                </g>
              </svg>
            </div>
          </template>
          <div v-else class="flex items-center justify-center h-full text-gray-500 text-sm">
            <Layers class="h-5 w-5 mr-2 opacity-50" />
            Select an image to preview annotations
          </div>
        </div>
      </div>

      <!-- Category legend -->
      <div v-if="categories.length > 0" class="flex flex-wrap gap-2 mt-2 pt-2 border-t border-background-secondary">
        <span
          v-for="cat in categories"
          :key="cat.id"
          class="inline-flex items-center gap-1 text-[11px] text-gray-400"
        >
          <span
            class="w-2.5 h-2.5 rounded-sm"
            :style="{ backgroundColor: categoryColor(cat.id) }"
          />
          {{ cat.name }}
        </span>
      </div>
    </div>
  </div>
</template>
