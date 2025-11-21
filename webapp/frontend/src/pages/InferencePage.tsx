import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { proteinsApi, inferenceApi } from '../services/api'
import type { InferenceJob } from '../types'

export default function InferencePage() {
  const [selectedProteinId, setSelectedProteinId] = useState<number | null>(null)
  const [extractAttention, setExtractAttention] = useState(true)
  const [activeJobs, setActiveJobs] = useState<InferenceJob[]>([])

  const { data: proteins } = useQuery({
    queryKey: ['proteins'],
    queryFn: proteinsApi.list,
  })

  const runInferenceMutation = useMutation({
    mutationFn: ({ proteinId, extractAttention }: { proteinId: number; extractAttention: boolean }) =>
      inferenceApi.run(proteinId, extractAttention),
    onSuccess: (job) => {
      setActiveJobs((prev) => [job, ...prev])
      setSelectedProteinId(null)
    },
  })

  const handleStartInference = async () => {
    if (!selectedProteinId) return

    await runInferenceMutation.mutateAsync({
      proteinId: selectedProteinId,
      extractAttention,
    })
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-white text-black border border-gray-800'
      case 'running':
        return 'bg-white text-black border border-gray-600'
      case 'failed':
        return 'bg-white text-black border border-gray-500'
      case 'pending':
        return 'bg-white text-black border border-gray-400'
      default:
        return 'bg-white text-black border border-gray-300'
    }
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-black">
        Run OpenFold Inference
      </h1>

      <div className="card">
        <h2 className="text-xl font-semibold mb-4 text-black">
          New Inference Job
        </h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2 text-gray-800">
              Select Protein
            </label>
            <select
              value={selectedProteinId || ''}
              onChange={(e) => setSelectedProteinId(Number(e.target.value))}
              className="input-field"
            >
              <option value="">Choose a protein...</option>
              {proteins?.map((protein) => (
                <option key={protein.id} value={protein.id}>
                  {protein.name} ({protein.sequence_length} aa)
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center">
            <input
              type="checkbox"
              id="extract-attention"
              checked={extractAttention}
              onChange={(e) => setExtractAttention(e.target.checked)}
              className="mr-2"
            />
            <label htmlFor="extract-attention" className="text-sm text-gray-800">
              Extract attention weights for visualization
            </label>
          </div>

          <button
            onClick={handleStartInference}
            disabled={!selectedProteinId || runInferenceMutation.isPending}
            className="btn-primary"
          >
            {runInferenceMutation.isPending ? 'Starting...' : 'Start Inference'}
          </button>

          {runInferenceMutation.error && (
            <div className="p-3 border border-gray-300 text-black rounded">
              Error: {runInferenceMutation.error.message}
            </div>
          )}
        </div>
      </div>

      {activeJobs.length > 0 && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4 text-black">
            Active Jobs
          </h2>

          <div className="space-y-3">
            {activeJobs.map((job) => (
              <div
                key={job.job_id}
                className="border border-gray-200 rounded-lg p-4"
              >
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <span className="font-medium text-black">
                      Job {job.job_id.slice(0, 8)}
                    </span>
                    <span
                      className={`ml-2 px-2 py-1 rounded text-xs uppercase tracking-wide ${getStatusColor(job.status)}`}
                    >
                      {job.status}
                    </span>
                  </div>
                  {job.progress > 0 && (
                    <span className="text-sm text-gray-700">
                      {job.progress.toFixed(1)}%
                    </span>
                  )}
                </div>

                {job.status === 'running' && (
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-black h-2 rounded-full transition-all"
                      style={{ width: `${job.progress}%` }}
                    />
                  </div>
                )}

                {job.error_message && (
                  <p className="mt-2 text-sm text-gray-800">
                    {job.error_message}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
