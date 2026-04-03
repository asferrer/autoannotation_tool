<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { getHealthStatus, listLabelingJobs, cancelLabelingJob } from '@/lib/api'
import { useTaskStore } from '@/stores/task'
import BaseButton from '@/components/ui/BaseButton.vue'
import {
  Plus, Activity, Cpu, CheckCircle, XCircle, PenSquare, Play,
  FolderOpen, X, Tags, Loader2, StopCircle, AlertCircle, RefreshCw,
} from 'lucide-vue-next'

const router = useRouter()
const taskStore = useTaskStore()
const gatewayOk = ref<boolean | null>(null)
const sam3Ok = ref<boolean | null>(null)
const jobs = ref<any[]>([])
let jobsInterval: ReturnType<typeof setInterval> | null = null

onMounted(async () => {
  try {
    const health = await getHealthStatus()
    gatewayOk.value = health.status === 'healthy'
    sam3Ok.value = health.services?.segmentation?.status === 'healthy'
  } catch {
    gatewayOk.value = false
    sam3Ok.value = false
  }
  await fetchJobs()
  // Poll only while there are running jobs
  jobsInterval = setInterval(async () => {
    await fetchJobs()
    const hasRunning = jobs.value.some((j) => j.status === 'running' || j.status === 'pending')
    if (!hasRunning && jobsInterval) {
      clearInterval(jobsInterval)
      jobsInterval = null
    }
  }, 3000)
})

onUnmounted(() => {
  if (jobsInterval) clearInterval(jobsInterval)
})

async function fetchJobs() {
  try {
    const result = await listLabelingJobs()
    // Show only last 5 jobs (most recent first)
    jobs.value = (result.jobs ?? []).slice(0, 5)
  } catch { /* gateway may be down */ }
}

async function cancelJob(jobId: string) {
  try {
    await cancelLabelingJob(jobId)
    await fetchJobs()
  } catch {}
}

function openJobInAnnotate(job: any) {
  const detectedClasses = Object.keys(job.objects_by_class ?? {})
  const labels = detectedClasses.map((name: string, i: number) => ({
    id: i + 1,
    name,
    color: taskStore.LABEL_COLORS[i % taskStore.LABEL_COLORS.length],
  }))
  const imageDir = job.image_directories?.[0] ?? ''
  const cocoPath = job.output_dir ? `${job.output_dir}/annotations.json` : ''
  taskStore.createTask(`Auto-labeled (${formatDate(job.created_at)})`, imageDir, cocoPath, labels)
  router.push('/annotate')
}

function resumeSession() { router.push('/annotate') }
function discardSession() { taskStore.clearSession() }

function formatDate(iso: string): string {
  if (!iso) return ''
  return new Date(iso).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
}

function statusColor(status: string) {
  if (status === 'completed') return 'text-green-400'
  if (status === 'running' || status === 'pending') return 'text-blue-400'
  if (status === 'failed') return 'text-red-400'
  if (status === 'cancelled') return 'text-gray-500'
  return 'text-gray-400'
}

function statusIcon(status: string) {
  if (status === 'completed') return CheckCircle
  if (status === 'running' || status === 'pending') return Loader2
  if (status === 'failed') return AlertCircle
  return X
}
</script>

<template>
  <div class="space-y-8">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white">Annotation Tool</h1>
        <p class="mt-1 text-gray-400">AI-powered dataset annotation with SAM3 support.</p>
      </div>
      <BaseButton @click="router.push('/create-task')" variant="primary" size="lg">
        <Plus class="h-5 w-5" /> New Task
      </BaseButton>
    </div>

    <!-- System Status -->
    <div class="grid gap-4 sm:grid-cols-2">
      <div :class="['card p-4 flex items-center gap-3', gatewayOk === true ? 'border-green-700/30' : gatewayOk === false ? 'border-red-700/30' : '']">
        <Activity class="h-5 w-5 text-primary" />
        <div class="flex-1">
          <p class="text-sm font-medium text-white">Gateway</p>
          <p class="text-xs text-gray-500">API service</p>
        </div>
        <CheckCircle v-if="gatewayOk === true" class="h-5 w-5 text-green-400" />
        <XCircle v-else-if="gatewayOk === false" class="h-5 w-5 text-red-400" />
        <div v-else class="h-5 w-5 rounded-full border-2 border-gray-600 border-t-gray-400 animate-spin" />
      </div>
      <div :class="['card p-4 flex items-center gap-3', sam3Ok === true ? 'border-green-700/30' : sam3Ok === false ? 'border-red-700/30' : '']">
        <Cpu class="h-5 w-5 text-primary" />
        <div class="flex-1">
          <p class="text-sm font-medium text-white">SAM3 Segmentation</p>
          <p class="text-xs text-gray-500">GPU inference</p>
        </div>
        <CheckCircle v-if="sam3Ok === true" class="h-5 w-5 text-green-400" />
        <XCircle v-else-if="sam3Ok === false" class="h-5 w-5 text-red-400" />
        <div v-else class="h-5 w-5 rounded-full border-2 border-gray-600 border-t-gray-400 animate-spin" />
      </div>
    </div>

    <!-- Resume annotation session -->
    <div v-if="taskStore.savedSessionInfo" class="card p-5 border border-primary/30 bg-primary/5">
      <div class="flex items-start justify-between gap-4">
        <div class="flex items-start gap-3 min-w-0">
          <Play class="h-5 w-5 text-primary mt-0.5 shrink-0" />
          <div class="min-w-0">
            <p class="text-sm font-semibold text-white">Resume: {{ taskStore.savedSessionInfo.config.name }}</p>
            <p class="text-xs text-gray-400 mt-0.5 flex items-center gap-1 truncate">
              <FolderOpen class="h-3 w-3 shrink-0" />
              {{ taskStore.savedSessionInfo.config.imagesDirPath }}
            </p>
            <p class="text-xs text-gray-500 mt-0.5">
              Image {{ taskStore.savedSessionInfo.lastImageIndex + 1 }} ·
              Saved {{ formatDate(taskStore.savedSessionInfo.savedAt) }}
            </p>
          </div>
        </div>
        <div class="flex items-center gap-2 shrink-0">
          <BaseButton @click="resumeSession" variant="primary" size="sm">
            <Play class="h-3.5 w-3.5" /> Resume
          </BaseButton>
          <button @click="discardSession" class="p-1.5 text-gray-500 hover:text-red-400 transition-colors" title="Discard session">
            <X class="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>

    <!-- Labeling jobs -->
    <div v-if="jobs.length > 0" class="card p-5 space-y-3">
      <div class="flex items-center justify-between">
        <h3 class="text-sm font-semibold text-gray-300 flex items-center gap-2">
          <Tags class="h-4 w-4 text-primary" /> Auto-Labeling Jobs
        </h3>
        <button @click="fetchJobs" class="text-gray-500 hover:text-gray-300 transition-colors" title="Refresh">
          <RefreshCw class="h-3.5 w-3.5" />
        </button>
      </div>

      <div class="space-y-2">
        <div
          v-for="job in jobs"
          :key="job.job_id"
          class="bg-background-tertiary rounded-lg px-3 py-2.5 flex items-center gap-3"
        >
          <!-- Status icon -->
          <component
            :is="statusIcon(job.status)"
            :class="['h-4 w-4 shrink-0', statusColor(job.status), (job.status === 'running' || job.status === 'pending') ? 'animate-spin' : '']"
          />

          <!-- Info -->
          <div class="flex-1 min-w-0">
            <p class="text-xs text-gray-300 truncate">
              {{ job.job_id.slice(0, 8) }}
              <span class="text-gray-500 ml-1">{{ formatDate(job.created_at) }}</span>
            </p>
            <div class="flex items-center gap-2 mt-0.5">
              <span :class="['text-xs font-medium', statusColor(job.status)]">{{ job.status }}</span>
              <span class="text-xs text-gray-500">
                {{ job.processed_images ?? 0 }}/{{ job.total_images ?? '?' }} img ·
                {{ job.annotations_created ?? 0 }} ann
              </span>
            </div>
            <!-- Progress bar for running jobs -->
            <div v-if="job.status === 'running'" class="mt-1 h-1 bg-gray-700 rounded-full overflow-hidden w-full">
              <div class="h-full bg-primary transition-all duration-500" :style="{ width: `${job.progress ?? 0}%` }" />
            </div>
          </div>

          <!-- Actions -->
          <div class="flex items-center gap-1 shrink-0">
            <button
              v-if="job.status === 'completed'"
              @click="openJobInAnnotate(job)"
              class="p-1.5 text-gray-400 hover:text-primary transition-colors"
              title="Open in Annotate"
            >
              <PenSquare class="h-3.5 w-3.5" />
            </button>
            <button
              v-if="job.status === 'running'"
              @click="cancelJob(job.job_id)"
              class="p-1.5 text-gray-400 hover:text-red-400 transition-colors"
              title="Cancel job"
            >
              <StopCircle class="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
      </div>

      <button
        @click="router.push('/jobs')"
        class="text-xs text-gray-500 hover:text-primary transition-colors"
      >
        View all jobs →
      </button>
    </div>

    <!-- Quick start -->
    <div class="card p-8 text-center">
      <PenSquare class="h-12 w-12 mx-auto mb-4 text-primary opacity-60" />
      <h2 class="text-xl font-semibold text-white mb-2">Start Annotating</h2>
      <p class="text-gray-400 text-sm mb-6 max-w-md mx-auto">
        Create a new task to load your images, configure labels, and start annotating manually or with AI-assisted auto-labeling.
      </p>
      <BaseButton @click="router.push('/create-task')" variant="primary">
        <Plus class="h-4 w-4" /> Create New Task
      </BaseButton>
    </div>
  </div>
</template>
