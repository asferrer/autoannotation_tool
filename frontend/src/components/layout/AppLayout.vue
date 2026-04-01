<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './Sidebar.vue'
import { useUiStore } from '@/stores/ui'

const route = useRoute()
const uiStore = useUiStore()

const isFullscreen = computed(() => route.meta.fullscreen === true)
</script>

<template>
  <div class="flex min-h-screen">
    <!-- Fullscreen mode: no sidebar, no padding -->
    <template v-if="isFullscreen">
      <main class="flex-1">
        <slot />
      </main>
    </template>

    <!-- Normal mode: sidebar + padded content -->
    <template v-else>
      <Sidebar />
      <main :class="['flex-1 transition-all duration-300', uiStore.sidebarCollapsed ? 'lg:ml-20' : 'lg:ml-64']">
        <div class="p-6 lg:p-8 max-w-7xl mx-auto">
          <slot />
        </div>
      </main>
    </template>

    <!-- Toast Notifications -->
    <div class="fixed bottom-4 right-4 z-[100] space-y-2">
      <div v-for="notification in uiStore.notifications" :key="notification.id"
        :class="['rounded-lg p-4 shadow-lg border max-w-sm',
          notification.type === 'success' ? 'bg-green-900/90 border-green-700 text-green-200' :
          notification.type === 'error' ? 'bg-red-900/90 border-red-700 text-red-200' :
          'bg-blue-900/90 border-blue-700 text-blue-200']">
        <p class="font-medium text-sm">{{ notification.title }}</p>
        <p class="text-xs mt-1 opacity-80">{{ notification.message }}</p>
      </div>
    </div>
  </div>
</template>
