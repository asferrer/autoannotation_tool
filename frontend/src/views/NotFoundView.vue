<script setup lang="ts">
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { Home, FileQuestion } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'

const router = useRouter()
const { t } = useI18n()
</script>

<template>
  <div class="flex flex-col items-center justify-center min-h-[60vh] text-center">
    <FileQuestion class="h-16 w-16 text-gray-500 mb-6" />
    <h1 class="text-3xl font-bold text-white mb-2">{{ t('notFound.title') }}</h1>
    <p class="text-gray-400 mb-8">{{ t('notFound.message') }}</p>
    <BaseButton @click="router.push('/')">
      <Home class="h-4 w-4 mr-2" />
      {{ t('notFound.goHome') }}
    </BaseButton>
  </div>
</template>
