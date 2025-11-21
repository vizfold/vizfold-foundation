export interface Protein {
  id: number
  name: string
  sequence: string
  sequence_length: number
  pdb_file?: string
  description?: string
  created_at: string
  updated_at: string
}

export interface InferenceJob {
  id: number
  job_id: string
  protein_id: number
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  started_at?: string
  completed_at?: string
  error_message?: string
  result_pdb?: string
  metadata?: Record<string, any>
  created_at: string
}

export interface Visualization {
  id: number
  protein_id: number
  viz_type: 'heatmap' | 'arc_diagram' | '3d' | 'combined'
  layer: number
  head?: number
  attention_type: 'msa_row' | 'triangle_start'
  residue_index?: number
  image_path: string
  thumbnail_path?: string
  metadata?: Record<string, any>
  created_at: string
}

export interface AttentionData {
  id: number
  protein_id: number
  layer: number
  head: number
  attention_type: string
  residue_index?: number
  data_file: string
  top_k?: number
}

export interface AttentionWeights {
  protein_id: number
  layer: number
  head: number
  attention_type: string
  weights: Array<{
    source: number
    target: number
    weight: number
  }>
}
