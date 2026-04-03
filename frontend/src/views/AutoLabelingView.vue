<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useUiStore } from '@/stores/ui'
import { useTaskStore } from '@/stores/task'
import {
  startLabeling,
  startRelabeling,
  getLabelingJobStatus,
  cancelLabelingJob,
  getLabelingJobPreviews,
  listDirectories,
} from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LiveAnnotationPreview from '@/components/labeling/LiveAnnotationPreview.vue'
import {
  Tags,
  Play,
  Plus,
  X,
  CheckCircle,
  RefreshCw,
  Sparkles,
  StopCircle,
  Fish,
  Car,
  Users,
  Building,
  TreePine,
  Utensils,
  Shirt,
  Trash2,
  PenSquare,
} from 'lucide-vue-next'
import type { LabelingJob, LabelingTaskType, RelabelMode } from '@/types/api'

interface LabelingPreview {
  filename: string
  image_data: string
}

const router = useRouter()
const uiStore = useUiStore()
const taskStore = useTaskStore()


// Predefined labeling templates
interface LabelingTemplate {
  id: string
  name: string
  desc: string
  icon: any
  classes: string[]
  taskType: LabelingTaskType
  confidence: number
}

const labelingTemplates: LabelingTemplate[] = [
  {
    id: 'marine_life',
    name: 'Marine Life',
    desc: 'Fish, coral, and ocean creatures',
    icon: Fish,
    classes: ['fish', 'coral', 'shark', 'turtle', 'jellyfish', 'octopus', 'starfish', 'crab', 'dolphin', 'whale'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'vehicles',
    name: 'Vehicles',
    desc: 'Cars, trucks, and transport',
    icon: Car,
    classes: ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'boat', 'airplane', 'train'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'people',
    name: 'People',
    desc: 'Persons and body parts',
    icon: Users,
    classes: ['person', 'face', 'hand', 'head'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'urban',
    name: 'Urban',
    desc: 'Buildings and street elements',
    icon: Building,
    classes: ['building', 'traffic light', 'stop sign', 'street sign', 'bench', 'parking meter', 'fire hydrant'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'nature',
    name: 'Nature',
    desc: 'Animals and plants',
    icon: TreePine,
    classes: ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'tree', 'flower'],
    taskType: 'both',
    confidence: 0.35,
  },
  {
    id: 'food',
    name: 'Food',
    desc: 'Food items and kitchenware',
    icon: Utensils,
    classes: ['apple', 'banana', 'orange', 'pizza', 'sandwich', 'cake', 'cup', 'bowl', 'bottle', 'knife', 'fork', 'spoon'],
    taskType: 'detection',
    confidence: 0.4,
  },
  {
    id: 'fashion',
    name: 'Fashion',
    desc: 'Clothing and accessories',
    icon: Shirt,
    classes: ['shirt', 'pants', 'dress', 'shoe', 'hat', 'bag', 'tie', 'watch', 'glasses', 'backpack'],
    taskType: 'detection',
    confidence: 0.35,
  },
  {
    id: 'marine_debris',
    name: 'Marine Debris',
    desc: 'Underwater waste and pollution',
    icon: Trash2,
    classes: ['Bottle', 'Can', 'Fishing_Net', 'Glove', 'Mask', 'Metal_Debris', 'Plastic_Debris', 'Tire'],
    taskType: 'both',
    confidence: 0.35,
  },
]

const selectedTemplate = ref<string | null>(null)

// Stable-ID wrappers for list items — prevents Vue reusing the wrong DOM node on splice
interface DirEntry { id: number; path: string }
interface ClassEntry { id: number; name: string }
let _nextId = 0
const mkDir = (path = ''): DirEntry => ({ id: _nextId++, path })
const mkClass = (name = ''): ClassEntry => ({ id: _nextId++, name })

function applyTemplate(template: LabelingTemplate) {
  selectedTemplate.value = template.id
  classes.value = template.classes.map(mkClass)
  taskType.value = template.taskType
  minConfidence.value = template.confidence
  uiStore.showSuccess(
    'Template Applied',
    `Applied "${template.name}" with ${template.classes.length} classes`
  )
}

function clearTemplate() {
  selectedTemplate.value = null
  classes.value = [mkClass()]
}

// Form state
const imageDirectories = ref<DirEntry[]>([mkDir()])
const classes = ref<ClassEntry[]>([mkClass()])
const outputDir = ref('/app/output/labeled')
const minConfidence = ref(0.3)
const taskType = ref<LabelingTaskType>('detection')
const outputFormats = ref<string[]>(['coco'])

// Preview mode state
const previewMode = ref(false)
const previewCount = ref(20)

// Deduplication strategy
const deduplicationStrategy = ref<'confidence' | 'area'>('confidence')

// Relabeling state
const isRelabeling = ref(false)
const relabelMode = ref<RelabelMode>('add')
const existingAnnotations = ref('')

// UI state
const loading = ref(false)
let pollingErrorCount = 0
const directories = ref<string[]>([])
const currentJob = ref<LabelingJob | null>(null)
const error = ref<string | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

// Preview state
const recentPreviews = ref<LabelingPreview[]>([])
const selectedPreview = ref<LabelingPreview | null>(null)
let previewPollingInterval: ReturnType<typeof setInterval> | null = null

const taskTypeOptions = computed(() => [
  { value: 'detection', label: 'Detection' },
  { value: 'segmentation', label: 'Segmentation' },
  { value: 'both', label: 'Both' },
])

const relabelModeOptions = computed(() => [
  { value: 'add', label: 'Add new annotations' },
  { value: 'replace', label: 'Replace existing' },
  { value: 'improve_segmentation', label: 'Improve segmentation' },
])

async function loadDirectories() {
  try {
    directories.value = await listDirectories('/')
  } catch (e) {
    // Ignore
  }
}

function addDirectory() {
  imageDirectories.value.push(mkDir())
}

function removeDirectory(index: number) {
  imageDirectories.value.splice(index, 1)
}

function addClass() {
  classes.value.push(mkClass())
}

function removeClass(index: number) {
  classes.value.splice(index, 1)
}

async function startJob() {
  const validDirs = imageDirectories.value.map((d: DirEntry) => d.path).filter((p: string) => p.trim())
  const validClasses = classes.value.map((c: ClassEntry) => c.name).filter((n: string) => n.trim())

  if (validDirs.length === 0) {
    uiStore.showError('Missing Input', 'Please add at least one image directory')
    return
  }

  if (!isRelabeling.value && validClasses.length === 0) {
    uiStore.showError('Missing Input', 'Please add at least one class to detect')
    return
  }

  if (isRelabeling.value && !existingAnnotations.value?.trim()) {
    uiStore.showError('Missing Input', 'Please provide the path to existing annotations')
    return
  }

  loading.value = true
  error.value = null
  recentPreviews.value = []
  selectedPreview.value = null

  try {
    let response

    if (isRelabeling.value) {
      response = await startRelabeling({
        image_directories: validDirs,
        output_dir: outputDir.value,
        relabel_mode: relabelMode.value,
        new_classes: validClasses.length > 0 ? validClasses : undefined,
        min_confidence: minConfidence.value,
        coco_json_path: existingAnnotations.value || undefined,
        output_formats: outputFormats.value,
        preview_mode: previewMode.value,
        preview_count: previewCount.value,
        deduplication_strategy: deduplicationStrategy.value,
      })
    } else {
      response = await startLabeling({
        image_directories: validDirs,
        classes: validClasses,
        output_dir: outputDir.value,
        min_confidence: minConfidence.value,
        task_type: taskType.value,
        output_formats: outputFormats.value,
        preview_mode: previewMode.value,
        preview_count: previewCount.value,
        deduplication_strategy: deduplicationStrategy.value,
      })
    }

    uiStore.showSuccess('Job Started', `Labeling job ${response.job_id.slice(0, 8)} is running`)
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || 'Failed to start labeling job'
    uiStore.showError('Job Failed', error.value ?? 'Unknown error')
  } finally {
    loading.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getLabelingJobStatus(jobId)
    currentJob.value = job
    pollingErrorCount = 0

    if (job.status === 'completed' || job.status === 'failed') {
      stopPolling()
      if (job.status === 'completed') {
        uiStore.showSuccess('Job Completed', `Processed ${job.processed_images} images`)
      } else {
        uiStore.showError('Job Failed', job.error || 'An unknown error occurred')
      }
    }
  } catch (e: any) {
    pollingErrorCount++
    if (pollingErrorCount >= 3) {
      uiStore.showInfo(
        'Polling Error',
        'Having trouble checking job status. Retrying...'
      )
      if (pollingErrorCount >= 10) {
        stopPolling()
        uiStore.showError('Connection Lost', 'Lost connection to the server. Please refresh.')
      }
    }
  }
}

async function pollPreviews(jobId: string) {
  try {
    const response = await getLabelingJobPreviews(jobId)
    recentPreviews.value = response.previews || []
  } catch (e) {
    // Ignore preview polling errors
  }
}

function startPolling(jobId: string) {
  pollingInterval = setInterval(() => pollJobStatus(jobId), 2000)
  // Poll previews less frequently (every 5 seconds)
  previewPollingInterval = setInterval(() => pollPreviews(jobId), 5000)
  // Initial preview load
  pollPreviews(jobId)
}

async function cancelCurrentJob() {
  if (!currentJob.value) return
  try {
    await cancelLabelingJob(currentJob.value.job_id)
    stopPolling()
    currentJob.value.status = 'cancelled'
    uiStore.showInfo('Job Cancelled', 'The labeling job has been cancelled')
  } catch (e: any) {
    uiStore.showError('Cancel Failed', e.message)
  }
}

function openInAnnotate() {
  if (!currentJob.value || currentJob.value.status !== 'completed') return
  const detectedClasses = Object.keys(currentJob.value.objects_by_class ?? {})
  const labels = detectedClasses.map((name, i) => ({
    id: i + 1,
    name,
    color: taskStore.LABEL_COLORS[i % taskStore.LABEL_COLORS.length],
  }))
  const imageDir = imageDirectories.value.filter((d: DirEntry) => d.path.trim())[0]?.path ?? ''
  // output_dir is the real subdirectory created by the backend (e.g. labeling_20240101_abc12345/)
  const cocoPath = currentJob.value.output_dir
    ? `${currentJob.value.output_dir}/annotations.json`
    : ''
  taskStore.createTask(`Auto-labeled (${new Date().toLocaleDateString()})`, imageDir, cocoPath, labels)
  router.push('/annotate')
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
  if (previewPollingInterval) {
    clearInterval(previewPollingInterval)
    previewPollingInterval = null
  }
  // Previews are intentionally kept visible after the job completes so the user
  // can inspect them. They are cleared only when a new job starts (startJob).
}

onMounted(() => {
  loadDirectories()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">Auto Labeling</h2>
      <p class="mt-1 text-gray-400">
        Automatically annotate your image datasets using AI models.
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Mode Toggle -->
    <div class="flex gap-4">
      <BaseButton
        :variant="!isRelabeling ? 'primary' : 'outline'"
        @click="isRelabeling = false"
      >
        <Tags class="h-5 w-5" />
        New Labeling
      </BaseButton>
      <BaseButton
        :variant="isRelabeling ? 'primary' : 'outline'"
        @click="isRelabeling = true"
      >
        <RefreshCw class="h-5 w-5" />
        Relabel Existing
      </BaseButton>
    </div>

    <!-- Templates Section (only for new labeling) -->
    <div v-if="!isRelabeling" class="card p-6">
      <div class="flex items-center justify-between mb-4">
        <div>
          <h3 class="text-lg font-semibold text-white flex items-center gap-2">
            <Sparkles class="h-5 w-5 text-yellow-400" />
            Quick Templates
          </h3>
          <p class="text-sm text-gray-400">Apply a preset configuration for common use cases</p>
        </div>
        <BaseButton
          v-if="selectedTemplate"
          variant="ghost"
          size="sm"
          @click="clearTemplate"
        >
          <X class="h-4 w-4" />
          Clear
        </BaseButton>
      </div>

      <div class="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <button
          v-for="template in labelingTemplates"
          :key="template.id"
          @click="applyTemplate(template)"
          :class="[
            'flex flex-col items-center gap-2 p-4 rounded-lg transition-all text-center',
            selectedTemplate === template.id
              ? 'bg-primary/20 border-2 border-primary'
              : 'bg-background-tertiary hover:bg-gray-600 border-2 border-transparent'
          ]"
        >
          <component :is="template.icon" class="h-8 w-8 text-primary" />
          <div>
            <p class="text-sm font-medium text-white">{{ template.name }}</p>
            <p class="text-xs text-gray-400">{{ template.desc }}</p>
          </div>
        </button>
      </div>
    </div>

    <div class="grid gap-6 lg:grid-cols-2">
      <!-- Input Configuration -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Image Directories</h3>

        <div class="space-y-3">
          <div
            v-for="(dir, index) in imageDirectories"
            :key="dir.id"
            class="flex gap-2 items-start"
          >
            <DirectoryBrowser
              v-model="dir.path"
              :label="index === 0 ? '' : undefined"
              placeholder="/app/datasets/images"
              path-mode="input"
              class="flex-1"
            />
            <BaseButton
              v-if="imageDirectories.length > 1"
              variant="ghost"
              size="sm"
              class="mt-8"
              @click="removeDirectory(index)"
            >
              <X class="h-4 w-4" />
            </BaseButton>
          </div>
          <BaseButton variant="outline" size="sm" @click="addDirectory">
            <Plus class="h-4 w-4" />
            Add Directory
          </BaseButton>
        </div>

        <!-- Classes (for new labeling) -->
        <div v-if="!isRelabeling" class="mt-6">
          <h4 class="text-md font-medium text-white mb-3">Classes to Detect</h4>
          <div class="space-y-3">
            <div
              v-for="(cls, index) in classes"
              :key="cls.id"
              class="flex gap-2"
            >
              <BaseInput
                v-model="cls.name"
                placeholder="e.g. car, person, dog"
                class="flex-1"
              />
              <BaseButton
                v-if="classes.length > 1"
                variant="ghost"
                size="sm"
                @click="removeClass(index)"
              >
                <X class="h-4 w-4" />
              </BaseButton>
            </div>
            <BaseButton variant="outline" size="sm" @click="addClass">
              <Plus class="h-4 w-4" />
              Add Class
            </BaseButton>
          </div>
        </div>

        <!-- Relabeling Options -->
        <div v-else class="mt-6 space-y-4">
          <BaseSelect
            v-model="relabelMode"
            :options="relabelModeOptions"
            label="Relabeling Mode"
          />
          <DirectoryBrowser
            v-model="existingAnnotations"
            label="Existing Annotations File"
            placeholder="/app/datasets/annotations.json"
            :show-files="true"
            file-pattern="*.json"
            path-mode="input"
            required
          />
        </div>
      </div>

      <!-- Output & Model Settings -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Output Settings</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="outputDir"
            label="Output Directory"
            placeholder="/app/output/labeled"
            path-mode="output"
          />

          <BaseSelect
            v-model="taskType"
            :options="taskTypeOptions"
            label="Task Type"
          />

          <div>
            <label class="text-sm text-gray-400 flex justify-between mb-2">
              <span>Min Confidence</span>
              <span class="text-white">{{ (minConfidence * 100).toFixed(0) }}%</span>
            </label>
            <input
              type="range"
              v-model.number="minConfidence"
              min="0.1"
              max="0.9"
              step="0.05"
              class="w-full accent-primary"
            />
          </div>

          <div>
            <label class="text-sm text-gray-400 mb-2 block">Output Formats</label>
            <div class="flex gap-4">
              <label
                v-for="fmt in ['coco', 'yolo', 'voc']"
                :key="fmt"
                class="flex items-center gap-2 cursor-pointer"
              >
                <input
                  type="checkbox"
                  :value="fmt"
                  v-model="outputFormats"
                  class="h-4 w-4 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                />
                <span class="text-gray-300 text-sm uppercase">{{ fmt }}</span>
              </label>
            </div>
          </div>

          <!-- Deduplication Strategy -->
          <div>
            <label class="text-sm text-gray-400 mb-2 block">Deduplication Strategy</label>
            <div class="flex gap-4">
              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  value="confidence"
                  v-model="deduplicationStrategy"
                  class="h-4 w-4 border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                />
                <div class="flex flex-col">
                  <span class="text-gray-300 text-sm">Confidence</span>
                  <span class="text-xs text-gray-500">Keep the most confident detection</span>
                </div>
              </label>
              <label class="flex items-center gap-2 cursor-pointer">
                <input
                  type="radio"
                  value="area"
                  v-model="deduplicationStrategy"
                  class="h-4 w-4 border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
                />
                <div class="flex flex-col">
                  <span class="text-gray-300 text-sm">Area</span>
                  <span class="text-xs text-gray-500">Keep the largest bounding box</span>
                </div>
              </label>
            </div>
          </div>

          <!-- Preview Mode -->
          <div class="border-t border-gray-700 pt-4">
            <div class="flex items-center gap-3 mb-3">
              <input
                type="checkbox"
                id="previewMode"
                v-model="previewMode"
                class="h-4 w-4 rounded border-gray-600 bg-background-tertiary text-primary focus:ring-primary"
              />
              <label for="previewMode" class="text-sm text-gray-300 cursor-pointer flex items-center gap-2">
                <Sparkles class="h-4 w-4 text-yellow-400" />
                Preview Mode
              </label>
            </div>
            <p class="text-xs text-gray-500 mb-3">Process only a subset of images to preview results before running the full job</p>
            <div v-if="previewMode">
              <label class="text-sm text-gray-400 flex justify-between mb-2">
                <span>Preview Count</span>
                <span class="text-white">{{ previewCount }} images</span>
              </label>
              <input
                type="range"
                v-model.number="previewCount"
                min="5"
                max="50"
                step="5"
                class="w-full accent-primary"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Current Job Progress -->
    <div v-if="currentJob" class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4">Job Progress</h3>

      <div class="flex items-center gap-4 mb-4">
        <component
          :is="currentJob.status === 'completed' ? CheckCircle : Tags"
          :class="[
            'h-8 w-8',
            currentJob.status === 'completed' ? 'text-green-400' :
            currentJob.status === 'failed' ? 'text-red-400' :
            'text-primary animate-pulse'
          ]"
        />
        <div class="flex-1">
          <p class="font-medium text-white">
            {{ currentJob.status === 'running' ? 'Processing...' : currentJob.status }}
          </p>
          <p class="text-sm text-gray-400">
            {{ currentJob.processed_images }} / {{ currentJob.total_images }} images |
            {{ currentJob.annotations_created }} annotations created
            <span v-if="currentJob.current_image && currentJob.status === 'running'" class="block text-xs text-gray-500 mt-0.5 truncate">
              Processing: {{ currentJob.current_image }}
            </span>
          </p>
        </div>
        <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
        <BaseButton
          v-if="currentJob.status === 'running'"
          variant="ghost"
          size="sm"
          @click="cancelCurrentJob"
        >
          <StopCircle class="h-4 w-4" />
          Cancel
        </BaseButton>
      </div>

      <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
        <div
          class="h-full bg-gradient-to-r from-primary to-green-400 transition-all duration-500"
          :style="{ width: `${currentJob.progress}%` }"
        />
      </div>

      <!-- Detections by Class -->
      <div v-if="currentJob.objects_by_class && Object.keys(currentJob.objects_by_class).length > 0" class="mt-4">
        <h4 class="text-sm font-medium text-gray-400 mb-2">Detections by Class</h4>
        <div class="flex flex-wrap gap-2">
          <div
            v-for="(count, className) in currentJob.objects_by_class"
            :key="className"
            class="bg-background-tertiary rounded-lg px-3 py-1.5 flex items-center gap-2"
          >
            <span class="text-sm text-gray-300">{{ className }}</span>
            <span class="text-sm font-semibold text-primary">{{ count }}</span>
          </div>
        </div>
      </div>

      <!-- Quality Metrics -->
      <div v-if="currentJob.quality_metrics" class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">Avg Confidence</p>
          <p class="text-lg font-semibold text-white">
            {{ (currentJob.quality_metrics.avg_confidence * 100).toFixed(1) }}%
          </p>
        </div>
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">With Detections</p>
          <p class="text-lg font-semibold text-green-400">
            {{ currentJob.quality_metrics.images_with_detections }}
          </p>
        </div>
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">No Detections</p>
          <p class="text-lg font-semibold text-yellow-400">
            {{ currentJob.quality_metrics.images_without_detections }}
          </p>
        </div>
        <div class="bg-background-tertiary rounded-lg p-3">
          <p class="text-xs text-gray-500 uppercase">Low Confidence</p>
          <p class="text-lg font-semibold text-orange-400">
            {{ currentJob.quality_metrics.low_confidence_count }}
          </p>
        </div>
      </div>

      <!-- Warnings -->
      <AlertBox
        v-if="currentJob.warnings && currentJob.warnings.length > 0"
        type="warning"
        title="Warnings"
        class="mt-4"
      >
        <ul class="list-disc list-inside text-sm">
          <li v-for="(warning, idx) in currentJob.warnings" :key="idx">{{ warning }}</li>
        </ul>
      </AlertBox>

      <!-- Error detail -->
      <AlertBox
        v-if="currentJob.status === 'failed' && currentJob.error"
        type="error"
        title="Job failed"
        class="mt-4"
      >
        <pre class="text-xs whitespace-pre-wrap break-words font-mono mt-1">{{ currentJob.error }}</pre>
      </AlertBox>

      <!-- Open in Annotate (when completed) -->
      <div v-if="currentJob.status === 'completed'" class="mt-4 flex justify-end">
        <BaseButton @click="openInAnnotate" variant="primary">
          <PenSquare class="h-4 w-4" />
          Open in Annotate
        </BaseButton>
      </div>

      <!-- Recent Previews -->
      <div v-if="recentPreviews.length > 0" class="mt-4">
        <h4 class="text-sm font-medium text-gray-400 mb-2">Recent Previews</h4>
        <div class="grid grid-cols-5 gap-2">
          <button
            v-for="preview in recentPreviews"
            :key="preview.filename"
            @click="selectedPreview = preview"
            class="relative aspect-square rounded overflow-hidden hover:ring-2 ring-primary transition-all"
          >
            <img :src="`data:image/jpeg;base64,${preview.image_data}`" class="w-full h-full object-cover" />
          </button>
        </div>
      </div>

      <!-- Live Annotation Preview -->
      <LiveAnnotationPreview
        v-if="currentJob && (currentJob.status === 'running' || currentJob.status === 'completed')"
        :job-id="currentJob.job_id"
        :is-running="currentJob.status === 'running'"
      />
    </div>

    <!-- Preview Modal -->
    <div
      v-if="selectedPreview"
      class="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
      @click="selectedPreview = null"
    >
      <div class="max-w-4xl max-h-[90vh] p-4" @click.stop>
        <img
          :src="`data:image/jpeg;base64,${selectedPreview.image_data}`"
          class="max-w-full max-h-[85vh] rounded-lg shadow-2xl"
        />
        <p class="text-center text-gray-400 mt-2">{{ selectedPreview.filename }}</p>
      </div>
    </div>

    <!-- Start Button -->
    <div class="flex justify-end">
      <BaseButton
        :loading="loading"
        :disabled="loading"
        @click="startJob"
        size="lg"
      >
        <Play class="h-5 w-5" />
        {{ isRelabeling ? 'Start Relabeling' : 'Start Labeling' }}
      </BaseButton>
    </div>
  </div>
</template>
