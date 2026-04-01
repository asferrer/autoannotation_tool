import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface TaskLabel {
  id: number
  name: string
  color: string
}

export interface TaskConfig {
  name: string
  imagesDirPath: string
  cocoJsonPath: string
  labels: TaskLabel[]
}

interface SavedSession {
  config: TaskConfig
  lastImageIndex: number
  savedAt: string
}

const LABEL_COLORS = [
  '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
  '#3b82f6', '#8b5cf6', '#ec4899', '#f43f5e', '#14b8a6',
  '#6366f1', '#84cc16', '#e879f9', '#fb923c', '#2dd4bf',
]

const STORAGE_KEY = 'ann_tool_session'

function loadFromStorage(): { config: TaskConfig | null; lastImageIndex: number } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { config: null, lastImageIndex: 0 }
    const saved: SavedSession = JSON.parse(raw)
    return { config: saved.config, lastImageIndex: saved.lastImageIndex ?? 0 }
  } catch {
    return { config: null, lastImageIndex: 0 }
  }
}

export const useTaskStore = defineStore('task', () => {
  const { config: savedConfig, lastImageIndex: savedIndex } = loadFromStorage()
  const config = ref<TaskConfig | null>(savedConfig)
  const lastSavedImageIndex = ref<number>(savedIndex)
  const isConfigured = computed(() => config.value !== null)

  const savedSessionInfo = computed<SavedSession | null>(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      return raw ? JSON.parse(raw) : null
    } catch { return null }
  })

  function createTask(name: string, imagesDirPath: string, cocoJsonPath: string, labels: TaskLabel[]) {
    config.value = { name, imagesDirPath, cocoJsonPath, labels }
    persistSession(0)
  }

  function persistSession(imageIndex: number) {
    if (!config.value) return
    try {
      const session: SavedSession = {
        config: config.value,
        lastImageIndex: imageIndex,
        savedAt: new Date().toISOString(),
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(session))
      lastSavedImageIndex.value = imageIndex
    } catch {}
  }

  function clearSession() {
    config.value = null
    lastSavedImageIndex.value = 0
    try { localStorage.removeItem(STORAGE_KEY) } catch {}
  }

  function getNextColor(): string {
    const usedCount = config.value?.labels.length ?? 0
    return LABEL_COLORS[usedCount % LABEL_COLORS.length]
  }

  function clear() {
    clearSession()
  }

  return {
    config, isConfigured, lastSavedImageIndex, savedSessionInfo,
    createTask, persistSession, clearSession, getNextColor, clear, LABEL_COLORS,
  }
})
