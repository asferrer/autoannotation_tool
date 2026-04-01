<script setup lang="ts">
import { watch } from 'vue'
import { useI18n } from 'vue-i18n'
import AppLayout from '@/components/layout/AppLayout.vue'
import { useSettingsStore } from '@/stores/settings'

const { locale } = useI18n()
const settings = useSettingsStore()

// Initialize locale from persisted settings
locale.value = settings.language

// Keep in sync when user changes language in Settings
watch(() => settings.language, (lang) => {
  locale.value = lang
})
</script>

<template>
  <AppLayout>
    <router-view />
  </AppLayout>
</template>
