<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useUiStore } from '@/stores/ui'
import { useSettingsStore } from '@/stores/settings'
import { getHealthStatus } from '@/lib/api'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import BaseSelect from '@/components/ui/BaseSelect.vue'
import {
  Server, CheckCircle, XCircle, Globe, Monitor, Info,
} from 'lucide-vue-next'

const { t, locale } = useI18n()
const uiStore = useUiStore()
const settings = useSettingsStore()

const healthStatus = ref<any>(null)
const checking = ref(false)

const languageOptions = [
  { value: 'en', label: 'English' },
  { value: 'es', label: 'Español' },
]

async function checkConnection() {
  checking.value = true
  try {
    healthStatus.value = await getHealthStatus()
  } catch {
    healthStatus.value = { status: 'error' }
  } finally {
    checking.value = false
  }
}

watch(() => settings.language, (lang) => {
  locale.value = lang
})

watch(() => settings.sidebarCollapsedDefault, (collapsed) => {
  if (collapsed !== uiStore.sidebarCollapsed) {
    uiStore.toggleSidebar()
  }
})

onMounted(() => {
  locale.value = settings.language
})
</script>

<template>
  <div class="space-y-6 max-w-2xl">
    <div>
      <h2 class="text-2xl font-bold text-white">{{ t('nav.settings') }}</h2>
      <p class="mt-2 text-gray-400">{{ t('settings.subtitle') }}</p>
    </div>

    <!-- Connection -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Server class="h-5 w-5 text-primary" />
        {{ t('settings.connection') }}
      </h3>
      <div class="space-y-4">
        <BaseInput
          v-model="settings.gatewayUrl"
          :label="t('settings.gatewayUrl')"
          placeholder="http://localhost:8000"
        />
        <div class="flex items-center gap-3">
          <BaseButton @click="checkConnection" :loading="checking" variant="outline">
            {{ t('settings.testConnection') }}
          </BaseButton>
          <div v-if="healthStatus" class="flex items-center gap-2">
            <CheckCircle v-if="healthStatus.status === 'healthy'" class="h-5 w-5 text-green-400" />
            <XCircle v-else class="h-5 w-5 text-red-400" />
            <span :class="healthStatus.status === 'healthy' ? 'text-green-300' : 'text-red-300'" class="text-sm">
              {{ healthStatus.status === 'healthy' ? t('settings.connected') : t('settings.connectionFailed') }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Language -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Globe class="h-5 w-5 text-primary" />
        {{ t('settings.language') }}
      </h3>
      <BaseSelect
        v-model="settings.language"
        :options="languageOptions"
        :label="t('settings.selectLanguage')"
      />
    </div>

    <!-- Display -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Monitor class="h-5 w-5 text-primary" />
        {{ t('settings.display') }}
      </h3>
      <label class="flex items-center gap-3 cursor-pointer">
        <input
          type="checkbox"
          v-model="settings.sidebarCollapsedDefault"
          class="w-4 h-4 rounded border-gray-600 bg-gray-700 text-primary focus:ring-primary"
        />
        <span class="text-sm text-gray-300">{{ t('settings.collapseSidebar') }}</span>
      </label>
    </div>

    <!-- About -->
    <div class="card p-6">
      <h3 class="text-lg font-semibold text-white mb-4 flex items-center gap-2">
        <Info class="h-5 w-5 text-primary" />
        {{ t('settings.about') }}
      </h3>
      <div class="space-y-2 text-sm text-gray-400">
        <p><span class="text-gray-300">{{ t('settings.version') }}:</span> 1.0.0</p>
        <p><span class="text-gray-300">{{ t('settings.stack') }}:</span> Vue 3 + FastAPI + SAM3</p>
      </div>
    </div>
  </div>
</template>
