import { ref, onUnmounted } from 'vue'

/**
 * Composable for polling an async function at a regular interval.
 * Automatically cleans up on component unmount.
 *
 * Usage:
 *   const { isPolling, start, stop } = usePolling(fetchStatus, 2000)
 *   start()
 */
export function usePolling(fn: () => Promise<void>, intervalMs: number, maxErrors = 10) {
  const isPolling = ref(false)
  const errorCount = ref(0)

  let timerId: ReturnType<typeof setInterval> | null = null

  async function poll() {
    try {
      await fn()
      errorCount.value = 0
    } catch {
      errorCount.value++
      if (errorCount.value >= maxErrors) {
        stop()
      }
    }
  }

  function start() {
    if (timerId) return
    isPolling.value = true
    errorCount.value = 0
    poll()
    timerId = setInterval(poll, intervalMs)
  }

  function stop() {
    if (timerId) {
      clearInterval(timerId)
      timerId = null
    }
    isPolling.value = false
  }

  onUnmounted(stop)

  return { isPolling, errorCount, start, stop }
}
