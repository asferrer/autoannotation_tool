import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUiStore = defineStore('ui', () => {
  const sidebarCollapsed = ref(false)
  const notifications = ref<{ id: string; type: string; title: string; message: string }[]>([])

  function toggleSidebar() {
    sidebarCollapsed.value = !sidebarCollapsed.value
  }

  function showSuccess(title: string, message: string) {
    addNotification('success', title, message)
  }

  function showError(title: string, message: string) {
    addNotification('error', title, message)
  }

  function showInfo(title: string, message: string) {
    addNotification('info', title, message)
  }

  function addNotification(type: string, title: string, message: string) {
    const id = Date.now().toString()
    notifications.value.push({ id, type, title, message })
    setTimeout(() => {
      notifications.value = notifications.value.filter(n => n.id !== id)
    }, 5000)
  }

  function dismissNotification(id: string) {
    notifications.value = notifications.value.filter(n => n.id !== id)
  }

  return {
    sidebarCollapsed,
    notifications,
    toggleSidebar,
    showSuccess,
    showError,
    showInfo,
    dismissNotification,
  }
})
