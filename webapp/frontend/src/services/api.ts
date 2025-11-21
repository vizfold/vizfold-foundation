import axios from 'axios'
import type { Protein, InferenceJob, Visualization, AttentionData, AttentionWeights } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Proteins API
export const proteinsApi = {
  list: async (): Promise<Protein[]> => {
    const response = await api.get('/api/proteins/')
    return response.data
  },

  get: async (id: number): Promise<Protein> => {
    const response = await api.get(`/api/proteins/${id}`)
    return response.data
  },

  create: async (data: { name: string; sequence: string; description?: string }): Promise<Protein> => {
    const response = await api.post('/api/proteins/', data)
    return response.data
  },

  upload: async (file: File, name: string, description?: string): Promise<Protein> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('name', name)
    if (description) formData.append('description', description)

    const response = await api.post('/api/proteins/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return response.data
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/api/proteins/${id}`)
  },

  getPdb: async (id: number): Promise<string> => {
    const response = await api.get(`/api/proteins/${id}/pdb`, {
      responseType: 'text',
    })
    return response.data
  },
}

// Inference API
export const inferenceApi = {
  run: async (proteinId: number, extractAttention = true): Promise<InferenceJob> => {
    const response = await api.post('/api/inference/run', {
      protein_id: proteinId,
      extract_attention: extractAttention,
    })
    return response.data
  },

  getStatus: async (jobId: string): Promise<InferenceJob> => {
    const response = await api.get(`/api/inference/${jobId}/status`)
    return response.data
  },

  getProteinJobs: async (proteinId: number): Promise<InferenceJob[]> => {
    const response = await api.get(`/api/inference/protein/${proteinId}`)
    return response.data
  },

  cancel: async (jobId: string): Promise<void> => {
    await api.delete(`/api/inference/${jobId}`)
  },
}

// Visualizations API
export const visualizationsApi = {
  generate: async (data: {
    protein_id: number
    viz_type: string
    layer: number
    head?: number
    attention_type: string
    residue_index?: number
    top_k?: number
  }): Promise<Visualization> => {
    const response = await api.post('/api/visualizations/generate', data)
    return response.data
  },

  getProteinVisualizations: async (
    proteinId: number,
    filters?: { viz_type?: string; layer?: number; attention_type?: string }
  ): Promise<Visualization[]> => {
    const response = await api.get(`/api/visualizations/protein/${proteinId}`, { params: filters })
    return response.data
  },

  get: async (vizId: number): Promise<Visualization> => {
    const response = await api.get(`/api/visualizations/${vizId}`)
    return response.data
  },

  delete: async (vizId: number): Promise<void> => {
    await api.delete(`/api/visualizations/${vizId}`)
  },
}

// Attention API
export const attentionApi = {
  getProteinAttentionData: async (
    proteinId: number,
    filters?: { layer?: number; head?: number; attention_type?: string }
  ): Promise<AttentionData[]> => {
    const response = await api.get(`/api/attention/protein/${proteinId}`, { params: filters })
    return response.data
  },

  getWeights: async (
    proteinId: number,
    layer: number,
    head: number,
    attentionType: string,
    residueIndex?: number
  ): Promise<AttentionWeights> => {
    const response = await api.get(`/api/attention/weights/${proteinId}/${layer}/${head}`, {
      params: { attention_type: attentionType, residue_index: residueIndex },
    })
    return response.data
  },

  getAvailableLayers: async (proteinId: number): Promise<number[]> => {
    const response = await api.get(`/api/attention/layers/${proteinId}`)
    return response.data.layers
  },

  getAvailableHeads: async (
    proteinId: number,
    layer: number,
    attentionType?: string
  ): Promise<number[]> => {
    const response = await api.get(`/api/attention/heads/${proteinId}/${layer}`, {
      params: { attention_type: attentionType },
    })
    return response.data.heads
  },

  download: async (
    proteinId: number,
    layer: number,
    head: number,
    attentionType: string,
    residueIndex?: number
  ): Promise<Blob> => {
    const response = await api.get(
      `/api/attention/download/${proteinId}/${layer}/${head}`,
      {
        params: { attention_type: attentionType, residue_index: residueIndex },
        responseType: 'blob',
      }
    )
    return response.data
  },
}

export default api
