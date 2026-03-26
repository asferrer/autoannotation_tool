<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { Tags, PenSquare, Box, Sparkles, Activity, Cpu, CheckCircle, XCircle, AlertTriangle } from 'lucide-vue-next'
import { getHealthStatus } from '@/lib/api'

const router = useRouter()
const { t } = useI18n()

const quickActions = [
  { label: t('home.startLabeling'), description: t('home.startLabelingDesc'), icon: Tags, path: '/auto-labeling', color: 'bg-blue-500/20 text-blue-400' },
  { label: t('home.reviewAnnotations'), description: t('home.reviewAnnotationsDesc'), icon: PenSquare, path: '/annotation-review', color: 'bg-green-500/20 text-green-400' },
  { label: t('home.manageLabels'), description: t('home.manageLabelsDesc'), icon: Sparkles, path: '/label-manager', color: 'bg-purple-500/20 text-purple-400' },
  { label: t('home.sam3Convert'), description: t('home.sam3ConvertDesc'), icon: Box, path: '/sam3-tools', color: 'bg-orange-500/20 text-orange-400' },
]

const systemStatus = ref<'loading' | 'healthy' | 'degraded' | 'error'>('loading')
const segmentationStatus = ref<string>('loading')

async function fetchStatus() {
  systemStatus.value = 'loading'
  try {
    const health = await getHealthStatus()
    systemStatus.value = health.status === 'healthy' ? 'healthy' : 'degraded'
    segmentationStatus.value = health.services?.segmentation?.status ?? 'unknown'
  } catch {
    systemStatus.value = 'error'
    segmentationStatus.value = 'unreachable'
  }
}

const statusConfig = {
  healthy: { icon: CheckCircle, color: 'text-green-400', bg: 'bg-green-500/10 border-green-700/30' },
  degraded: { icon: AlertTriangle, color: 'text-yellow-400', bg: 'bg-yellow-500/10 border-yellow-700/30' },
  error: { icon: XCircle, color: 'text-red-400', bg: 'bg-red-500/10 border-red-700/30' },
  loading: { icon: Activity, color: 'text-gray-400', bg: 'bg-gray-500/10 border-gray-700/30' },
}

onMounted(fetchStatus)
</script>

<template>
  <div class="space-y-8">
    <div>
      <h1 class="text-3xl font-bold text-white">{{ t('home.welcome') }}</h1>
      <p class="mt-2 text-gray-400">{{ t('home.subtitle') }}</p>
    </div>

    <!-- Quick Actions -->
    <div>
      <h2 class="text-lg font-semibold text-white mb-4">{{ t('home.quickActions') }}</h2>
      <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <button
          v-for="action in quickActions"
          :key="action.path"
          @click="router.push(action.path)"
          class="card p-6 text-left hover:border-primary/50 transition-all group"
        >
          <div :class="['rounded-lg p-3 w-fit mb-4', action.color]">
            <component :is="action.icon" class="h-6 w-6" />
          </div>
          <h3 class="font-semibold text-white group-hover:text-primary transition-colors">{{ action.label }}</h3>
          <p class="text-sm text-gray-400 mt-1">{{ action.description }}</p>
        </button>
      </div>
    </div>

    <!-- System Status -->
    <div>
      <h2 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Activity class="h-5 w-5 text-gray-400" />
        {{ t('home.systemStatus') }}
      </h2>
      <div class="grid gap-4 sm:grid-cols-2">
        <!-- Gateway -->
        <div :class="['card p-5 border', statusConfig[systemStatus].bg]">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <Activity class="h-5 w-5 text-primary" />
              <span class="font-medium text-white">Gateway</span>
            </div>
            <div class="flex items-center gap-2">
              <component :is="statusConfig[systemStatus].icon" :class="['h-4 w-4', statusConfig[systemStatus].color]" />
              <span :class="['text-sm', statusConfig[systemStatus].color]">{{ t(`home.status.${systemStatus}`) }}</span>
            </div>
          </div>
        </div>

        <!-- Segmentation -->
        <div :class="['card p-5 border', segmentationStatus === 'healthy' ? statusConfig.healthy.bg : segmentationStatus === 'unreachable' ? statusConfig.error.bg : statusConfig.degraded.bg]">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              <Cpu class="h-5 w-5 text-primary" />
              <span class="font-medium text-white">SAM3 Segmentation</span>
            </div>
            <div class="flex items-center gap-2">
              <CheckCircle v-if="segmentationStatus === 'healthy'" class="h-4 w-4 text-green-400" />
              <XCircle v-else class="h-4 w-4 text-red-400" />
              <span :class="['text-sm', segmentationStatus === 'healthy' ? 'text-green-400' : 'text-red-400']">
                {{ t(`home.status.${segmentationStatus === 'healthy' ? 'healthy' : 'error'}`) }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
