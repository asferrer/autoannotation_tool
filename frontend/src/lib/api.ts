import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

// Health
export async function getHealthStatus() {
  const { data } = await api.get('/health')
  return data
}

// Annotations CRUD
export async function loadAnnotations(cocoJsonPath: string) {
  const { data } = await api.post('/annotations/load', { coco_json_path: cocoJsonPath })
  return data
}

export async function saveAnnotations(request: any) {
  const { data } = await api.post('/annotations/save', request)
  return data
}

export async function getImageAnnotations(imageId: number, cocoJsonPath: string) {
  const { data } = await api.get('/annotations/images/' + imageId, { params: { coco_json_path: cocoJsonPath } })
  return data
}

// Labeling
export async function startLabeling(request: any) {
  const { data } = await api.post('/labeling/start', request)
  return data
}

export async function startRelabeling(request: any) {
  const { data } = await api.post('/labeling/relabel', request)
  return data
}

export async function getLabelingJobStatus(jobId: string) {
  const { data } = await api.get(`/labeling/jobs/${jobId}`)
  return data
}

export async function cancelLabelingJob(jobId: string) {
  const { data } = await api.delete(`/labeling/jobs/${jobId}`)
  return data
}

export async function getLabelingJobResult(jobId: string) {
  const { data } = await api.get(`/labeling/jobs/${jobId}/result`)
  return data
}

export async function getLabelingJobPreviews(jobId: string) {
  const { data } = await api.get(`/labeling/jobs/${jobId}/previews`)
  return data
}

// SAM3
export async function segmentWithText(imagePath: string, textPrompt: string) {
  const { data } = await api.post('/sam3/segment-image', { image_path: imagePath, text_prompt: textPrompt })
  return data
}

export async function sam3ConvertDataset(request: any) {
  const { data } = await api.post('/sam3/convert-dataset', request)
  return data
}

export async function getSam3JobStatus(jobId: string) {
  const { data } = await api.get(`/sam3/jobs/${jobId}`)
  return data
}

// Datasets
export async function listDatasets() {
  const { data } = await api.get('/datasets')
  return data
}

export async function analyzeDataset(datasetPath: string) {
  const { data } = await api.post('/datasets/analyze', { dataset_path: datasetPath })
  return data
}

export async function renameCategory(datasetPath: string, categoryId: number, newName: string) {
  const { data } = await api.post('/datasets/categories/rename', { dataset_path: datasetPath, category_id: categoryId, new_name: newName })
  return data
}

export async function deleteCategory(datasetPath: string, categoryId: number) {
  const { data } = await api.delete(`/datasets/categories/${categoryId}`, { params: { dataset_path: datasetPath } })
  return data
}

// Filesystem
export async function listDirectories(path: string) {
  const { data } = await api.get('/filesystem/browse', { params: { path, type: 'directories' } })
  return data
}

export async function listFiles(path: string, pattern?: string) {
  const { data } = await api.get('/filesystem/browse', { params: { path, type: 'files', pattern } })
  return data
}

export async function checkPathExists(path: string): Promise<{ exists: boolean; is_directory: boolean }> {
  try {
    const { data } = await api.get('/filesystem/check-path', { params: { path } })
    return data
  } catch {
    return { exists: false, is_directory: false }
  }
}

export function getImageUrl(imagePath: string): string {
  return `/api/filesystem/image?path=${encodeURIComponent(imagePath)}`
}

// Jobs
export async function getAllJobs() {
  const { data } = await api.get('/jobs')
  return data
}

export default api
