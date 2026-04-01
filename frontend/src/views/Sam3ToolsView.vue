<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import { useUiStore } from '@/stores/ui'
import {
  segmentWithText,
  sam3ConvertDataset,
  getSam3JobStatus,
} from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import AlertBox from '@/components/common/AlertBox.vue'
import LoadingSpinner from '@/components/common/LoadingSpinner.vue'
import DirectoryBrowser from '@/components/common/DirectoryBrowser.vue'
import {
  Box,
  Wand2,
  Database,
  CheckCircle,
  Play,
} from 'lucide-vue-next'
import type { SegmentationResult, Job } from '@/types/api'

const uiStore = useUiStore()

// Mode
const mode = ref<'text' | 'convert'>('text')

// Text segmentation state
const imagePath = ref('')
const textPrompt = ref('')
const textLoading = ref(false)
const textResult = ref<{ success: boolean; segmentation_coco: number[][] | null; bbox: number[] | null; confidence: number; error?: string } | null>(null)

// Dataset conversion state
const cocoJsonPath = ref('')
const imagesDir = ref('')
const outputDir = ref('/app/output/sam3_converted')
const minArea = ref(100)
const confidenceThreshold = ref(0.8)
const convertLoading = ref(false)
const currentJob = ref<Job | null>(null)
let pollingInterval: ReturnType<typeof setInterval> | null = null

const error = ref<string | null>(null)

async function runTextSegmentation() {
  if (!imagePath.value || !textPrompt.value) {
    uiStore.showError('Missing Input', 'Please provide an image path and a text prompt')
    return
  }

  textLoading.value = true
  error.value = null
  textResult.value = null

  try {
    textResult.value = await segmentWithText(imagePath.value, textPrompt.value)
    uiStore.showSuccess(
      'Segmentation Complete',
      textResult.value?.success ? 'Mask generated' : (textResult.value?.error ?? 'No mask found')
    )
  } catch (e: any) {
    error.value = e.message || 'Segmentation failed'
    uiStore.showError('Segmentation Failed', error.value ?? 'An unknown error occurred')
  } finally {
    textLoading.value = false
  }
}

async function startConversion() {
  if (!cocoJsonPath.value || !imagesDir.value || !outputDir.value) {
    uiStore.showError('Missing Input', 'Please fill in all required fields')
    return
  }

  convertLoading.value = true
  error.value = null
  currentJob.value = null

  try {
    const response = await sam3ConvertDataset({
      coco_json_path: cocoJsonPath.value,
      images_dir: imagesDir.value,
      output_dir: outputDir.value,
      min_area: minArea.value,
      confidence_threshold: confidenceThreshold.value,
    })

    uiStore.showSuccess(
      'Conversion Started',
      `Job ${response.job_id.slice(0, 8)} is running`
    )
    startPolling(response.job_id)
  } catch (e: any) {
    error.value = e.message || 'Conversion failed'
    uiStore.showError('Conversion Failed', error.value ?? 'An unknown error occurred')
    convertLoading.value = false
  }
}

async function pollJobStatus(jobId: string) {
  try {
    const job = await getSam3JobStatus(jobId)
    currentJob.value = job

    if (job.status === 'completed') {
      stopPolling()
      convertLoading.value = false
      uiStore.showSuccess('Conversion Complete', 'Dataset has been converted successfully')
    } else if (job.status === 'failed') {
      stopPolling()
      convertLoading.value = false
      error.value = job.error || 'Conversion failed'
      uiStore.showError('Conversion Failed', error.value ?? 'An unknown error occurred')
    }
  } catch (e) {
    // Ignore polling errors
  }
}

function startPolling(jobId: string) {
  pollingInterval = setInterval(() => pollJobStatus(jobId), 2000)
}

function stopPolling() {
  if (pollingInterval) {
    clearInterval(pollingInterval)
    pollingInterval = null
  }
}

onUnmounted(() => {
  stopPolling()
})
</script>

<template>
  <div class="space-y-6">
    <!-- Header -->
    <div>
      <h2 class="text-2xl font-bold text-white">SAM3 Segmentation Tools</h2>
      <p class="mt-1 text-gray-400">
        Use SAM3 to segment objects by text prompt or convert detection datasets to segmentation format.
      </p>
    </div>

    <!-- Error -->
    <AlertBox v-if="error" type="error" :title="error" dismissible @dismiss="error = null" />

    <!-- Mode Toggle -->
    <div class="flex gap-2">
      <BaseButton
        :variant="mode === 'text' ? 'primary' : 'outline'"
        @click="mode = 'text'"
      >
        <Wand2 class="h-5 w-5" />
        Text Segmentation
      </BaseButton>
      <BaseButton
        :variant="mode === 'convert' ? 'primary' : 'outline'"
        @click="mode = 'convert'"
      >
        <Database class="h-5 w-5" />
        Convert Dataset
      </BaseButton>
    </div>

    <!-- Text Segmentation Mode -->
    <div v-if="mode === 'text'" class="grid gap-6 lg:grid-cols-2">
      <!-- Input -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Text-Guided Segmentation</h3>

        <div class="space-y-4">
          <DirectoryBrowser
            v-model="imagePath"
            label="Image Path"
            placeholder="/app/datasets/images/example.jpg"
            :show-files="true"
            file-pattern="*.jpg,*.png,*.jpeg"
            path-mode="input"
          />
          <BaseInput
            v-model="textPrompt"
            label="Text Prompt"
            placeholder="e.g. fish, plastic bottle, person"
            hint="Describe the objects you want to segment"
          />
        </div>

        <BaseButton
          class="mt-6 w-full"
          :loading="textLoading"
          :disabled="textLoading || !imagePath || !textPrompt"
          @click="runTextSegmentation"
        >
          <Wand2 class="h-5 w-5" />
          Run Segmentation
        </BaseButton>
      </div>

      <!-- Results -->
      <div class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Results</h3>

        <div v-if="textLoading" class="flex justify-center py-12">
          <LoadingSpinner message="Processing..." />
        </div>

        <div v-else-if="!textResult" class="text-center py-12 text-gray-400">
          <Box class="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Run segmentation to see results</p>
        </div>

        <div v-else class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-gray-400">Result</span>
            <span class="text-2xl font-bold text-primary">{{ textResult.success ? '1 mask' : 'None' }}</span>
          </div>

          <div class="space-y-2">
            <div v-if="textResult.success" class="flex items-center gap-3">
              <span class="w-20 text-sm text-gray-400">Confidence</span>
              <div class="flex-1 h-2 bg-background-tertiary rounded-full overflow-hidden">
                <div
                  class="h-full bg-primary"
                  :style="{ width: `${(textResult.confidence ?? 0) * 100}%` }"
                />
              </div>
              <span class="text-sm text-white">{{ ((textResult.confidence ?? 0) * 100).toFixed(1) }}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Dataset Conversion Mode -->
    <div v-if="mode === 'convert'" class="space-y-6">
      <div class="grid gap-6 lg:grid-cols-2">
        <!-- Input -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-white mb-4">Source Dataset</h3>
          <p class="text-sm text-gray-400 mb-4">
            Convert a COCO detection dataset to segmentation format using SAM3 to generate masks.
          </p>

          <div class="space-y-4">
            <DirectoryBrowser
              v-model="cocoJsonPath"
              label="COCO JSON Path"
              placeholder="/app/datasets/annotations.json"
              :show-files="true"
              file-pattern="*.json"
              path-mode="input"
            />
            <DirectoryBrowser
              v-model="imagesDir"
              label="Images Directory"
              placeholder="/app/datasets/images"
              path-mode="input"
            />
          </div>
        </div>

        <!-- Output -->
        <div class="card p-6">
          <h3 class="text-lg font-semibold text-white mb-4">Output Settings</h3>

          <div class="space-y-4">
            <DirectoryBrowser
              v-model="outputDir"
              label="Output Directory"
              placeholder="/app/output/sam3_converted"
              path-mode="output"
            />

            <div>
              <label class="text-sm text-gray-400 flex justify-between mb-2">
                <span>Min Mask Area</span>
                <span class="text-white">{{ minArea }} px²</span>
              </label>
              <input
                type="range"
                v-model.number="minArea"
                min="10"
                max="1000"
                step="10"
                class="w-full accent-primary"
              />
              <p class="text-xs text-gray-500 mt-1">Masks smaller than this area will be discarded</p>
            </div>

            <div>
              <label class="text-sm text-gray-400 flex justify-between mb-2">
                <span>Confidence Threshold</span>
                <span class="text-white">{{ (confidenceThreshold * 100).toFixed(0) }}%</span>
              </label>
              <input
                type="range"
                v-model.number="confidenceThreshold"
                min="0.1"
                max="1"
                step="0.05"
                class="w-full accent-primary"
              />
              <p class="text-xs text-gray-500 mt-1">Minimum SAM3 confidence score to accept a mask</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Conversion Progress -->
      <div v-if="currentJob" class="card p-6">
        <h3 class="text-lg font-semibold text-white mb-4">Conversion Progress</h3>

        <div class="flex items-center gap-4 mb-4">
          <component
            :is="currentJob.status === 'completed' ? CheckCircle : Database"
            :class="[
              'h-8 w-8',
              currentJob.status === 'completed' ? 'text-green-400' : 'text-primary animate-pulse'
            ]"
          />
          <div class="flex-1">
            <p class="font-medium text-white">
              {{ currentJob.status === 'running' ? 'Converting...' : currentJob.status }}
            </p>
            <p class="text-sm text-gray-400">Job {{ currentJob.job_id.slice(0, 8) }}...</p>
          </div>
          <span class="text-2xl font-bold text-primary">{{ currentJob.progress }}%</span>
        </div>

        <div class="h-3 bg-background-tertiary rounded-full overflow-hidden">
          <div
            class="h-full bg-primary transition-all"
            :style="{ width: `${currentJob.progress}%` }"
          />
        </div>
      </div>

      <!-- Start Button -->
      <div class="flex justify-end">
        <BaseButton
          :loading="convertLoading"
          :disabled="convertLoading || !cocoJsonPath || !imagesDir || !outputDir"
          @click="startConversion"
          size="lg"
        >
          <Play class="h-5 w-5" />
          Start Conversion
        </BaseButton>
      </div>
    </div>
  </div>
</template>
