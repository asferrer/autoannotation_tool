import { ref, computed, onMounted } from 'vue'
import { checkPathExists } from '@/lib/api'

export interface MountPoint {
  id: string
  name: string
  path: string
  description: string
  icon: string
  exists: boolean
  purpose: 'input' | 'output' | 'both'
}

const DEFAULT_MOUNT_POINTS: Omit<MountPoint, 'exists'>[] = [
  {
    id: 'datasets',
    name: 'Datasets',
    path: '/app/datasets',
    description: 'Dataset files (COCO JSON, images)',
    icon: 'database',
    purpose: 'both',
  },
  {
    id: 'output',
    name: 'Output',
    path: '/app/output',
    description: 'Output directory for labeling results',
    icon: 'folder-output',
    purpose: 'output',
  },
]

const mountPoints = ref<MountPoint[]>([])
const loading = ref(true)
const loaded = ref(false)

let initialized = false

async function initialize() {
  if (initialized) return
  initialized = true

  loading.value = true
  const results: MountPoint[] = []

  for (const mp of DEFAULT_MOUNT_POINTS) {
    let exists = false
    try {
      const result = await checkPathExists(mp.path)
      exists = result.exists && result.is_directory
    } catch {
      exists = false
    }
    results.push({ ...mp, exists })
  }

  mountPoints.value = results
  loading.value = false
  loaded.value = true
}

export function useMountPoints() {
  onMounted(() => {
    initialize()
  })

  function getFilteredMountPoints(mode: 'input' | 'output' | 'both'): MountPoint[] {
    if (mode === 'both') return mountPoints.value
    return mountPoints.value.filter(
      (mp) => mp.purpose === mode || mp.purpose === 'both'
    )
  }

  function getDefaultPath(mode: 'input' | 'output' | 'both'): string | null {
    const filtered = getFilteredMountPoints(mode)
    const existing = filtered.find((mp) => mp.exists)
    return existing?.path ?? filtered[0]?.path ?? null
  }

  function isPathValid(path: string, mode: 'input' | 'output' | 'both'): boolean {
    const filtered = getFilteredMountPoints(mode)
    return filtered.some((mp) => path === mp.path || path.startsWith(mp.path + '/'))
  }

  function getMountPointForPath(path: string): MountPoint | undefined {
    return mountPoints.value.find(
      (mp) => path === mp.path || path.startsWith(mp.path + '/')
    )
  }

  return {
    mountPoints: computed(() => mountPoints.value),
    loading: computed(() => loading.value),
    loaded: computed(() => loaded.value),
    getFilteredMountPoints,
    getDefaultPath,
    isPathValid,
    getMountPointForPath,
  }
}
