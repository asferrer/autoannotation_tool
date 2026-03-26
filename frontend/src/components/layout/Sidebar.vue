<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import {
  LayoutDashboard,
  Tags,
  PenSquare,
  Box,
  Settings,
  ChevronLeft,
  ChevronRight,
  Sparkles,
} from 'lucide-vue-next'

const route = useRoute()
const uiStore = useUiStore()
const { t } = useI18n()

const annotateItems = [
  { label: t('nav.autoLabeling'), path: '/auto-labeling', icon: Tags },
  { label: t('nav.annotationReview'), path: '/annotation-review', icon: PenSquare },
]

const manageItems = [
  { label: t('nav.labelManager'), path: '/label-manager', icon: Tags },
  { label: t('nav.sam3Tools'), path: '/sam3-tools', icon: Box },
]

const isActive = (path: string) => route.path === path

const sidebarClasses = computed(() => [
  'fixed inset-y-0 left-0 z-50 flex flex-col bg-background-secondary border-r border-gray-700/50',
  'transition-all duration-300 ease-in-out',
  uiStore.sidebarCollapsed ? 'w-20' : 'w-64',
])
</script>

<template>
  <aside :class="sidebarClasses" class="hidden lg:flex">
    <!-- Logo -->
    <div class="flex h-16 items-center justify-between border-b border-gray-700/50 px-4">
      <router-link to="/" class="flex items-center gap-3">
        <div class="flex h-10 w-10 items-center justify-center rounded-xl bg-primary">
          <PenSquare class="h-6 w-6 text-white" />
        </div>
        <span v-if="!uiStore.sidebarCollapsed" class="font-semibold text-white">
          {{ t('app.shortName') }}
        </span>
      </router-link>
    </div>

    <nav class="flex-1 overflow-y-auto px-3 py-4">
      <!-- Dashboard -->
      <div class="mb-6">
        <router-link
          to="/"
          :class="[
            'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
            isActive('/') ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
          ]"
        >
          <LayoutDashboard class="h-5 w-5 flex-shrink-0" />
          <span v-if="!uiStore.sidebarCollapsed">{{ t('nav.dashboard') }}</span>
        </router-link>
      </div>

      <!-- Annotate -->
      <div class="mb-6">
        <h3 v-if="!uiStore.sidebarCollapsed" class="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
          {{ t('nav.annotate') }}
        </h3>
        <div class="space-y-1">
          <router-link
            v-for="item in annotateItems"
            :key="item.path"
            :to="item.path"
            :class="[
              'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
              isActive(item.path) ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
            ]"
          >
            <component :is="item.icon" class="h-5 w-5 flex-shrink-0" />
            <span v-if="!uiStore.sidebarCollapsed">{{ item.label }}</span>
          </router-link>
        </div>
      </div>

      <!-- Manage -->
      <div class="mb-6">
        <h3 v-if="!uiStore.sidebarCollapsed" class="mb-2 px-3 text-xs font-semibold uppercase tracking-wider text-gray-500">
          {{ t('nav.manage') }}
        </h3>
        <div class="space-y-1">
          <router-link
            v-for="item in manageItems"
            :key="item.path"
            :to="item.path"
            :class="[
              'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
              isActive(item.path) ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
            ]"
          >
            <component :is="item.icon" class="h-5 w-5 flex-shrink-0" />
            <span v-if="!uiStore.sidebarCollapsed">{{ item.label }}</span>
          </router-link>
        </div>
      </div>

      <!-- Settings -->
      <div>
        <router-link
          to="/settings"
          :class="[
            'flex items-center gap-3 rounded-lg px-3 py-2.5 transition-colors',
            isActive('/settings') ? 'bg-primary text-white' : 'text-gray-400 hover:bg-gray-700/50 hover:text-white',
          ]"
        >
          <Settings class="h-5 w-5 flex-shrink-0" />
          <span v-if="!uiStore.sidebarCollapsed">{{ t('nav.settings') }}</span>
        </router-link>
      </div>
    </nav>

    <!-- Collapse -->
    <div class="border-t border-gray-700/50 p-3">
      <button
        @click="uiStore.toggleSidebar"
        class="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-gray-400 hover:bg-gray-700/50 hover:text-white transition-colors"
      >
        <ChevronLeft v-if="!uiStore.sidebarCollapsed" class="h-5 w-5" />
        <ChevronRight v-else class="h-5 w-5" />
        <span v-if="!uiStore.sidebarCollapsed">{{ t('common.actions.collapse') }}</span>
      </button>
    </div>
  </aside>
</template>
