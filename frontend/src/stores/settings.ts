import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

const STORAGE_KEY = 'annotation-tool-settings'

export const useSettingsStore = defineStore('settings', () => {
  const gatewayUrl = ref('http://localhost:8000')
  const language = ref('en')
  const sidebarCollapsedDefault = ref(false)

  function load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) {
        const saved = JSON.parse(raw)
        gatewayUrl.value = saved.gatewayUrl ?? gatewayUrl.value
        language.value = saved.language ?? language.value
        sidebarCollapsedDefault.value = saved.sidebarCollapsedDefault ?? sidebarCollapsedDefault.value
      }
    } catch {
      // Ignore corrupted data
    }
  }

  function save() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      gatewayUrl: gatewayUrl.value,
      language: language.value,
      sidebarCollapsedDefault: sidebarCollapsedDefault.value,
    }))
  }

  // Auto-save on changes
  watch([gatewayUrl, language, sidebarCollapsedDefault], save, { deep: true })

  // Load on creation
  load()

  return {
    gatewayUrl,
    language,
    sidebarCollapsedDefault,
    load,
    save,
  }
})
