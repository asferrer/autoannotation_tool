<script setup lang="ts">
withDefaults(defineProps<{
  lines?: number
  variant?: 'text' | 'card' | 'image'
}>(), {
  lines: 3,
  variant: 'text',
})
</script>

<template>
  <div v-if="variant === 'text'" class="space-y-3 animate-pulse">
    <div
      v-for="i in lines"
      :key="i"
      class="h-4 bg-gray-700 rounded"
      :style="{ width: i === lines ? '60%' : '100%' }"
    />
  </div>

  <div v-else-if="variant === 'card'" class="card p-6 animate-pulse">
    <div class="h-5 bg-gray-700 rounded w-1/3 mb-4" />
    <div class="space-y-2">
      <div class="h-4 bg-gray-700 rounded w-full" />
      <div class="h-4 bg-gray-700 rounded w-4/5" />
    </div>
  </div>

  <div v-else-if="variant === 'image'" class="animate-pulse">
    <div class="bg-gray-700 rounded-lg aspect-video w-full" />
  </div>
</template>
