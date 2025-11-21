import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { proteinsApi } from '../services/api'
import type { Protein } from '../types'
import ProteinUploadModal from '../components/ProteinUploadModal'

export default function ProteinsPage() {
  const [showUploadModal, setShowUploadModal] = useState(false)
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const { data: proteins, isLoading } = useQuery({
    queryKey: ['proteins'],
    queryFn: proteinsApi.list,
  })

  const deleteMutation = useMutation({
    mutationFn: proteinsApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['proteins'] })
    },
  })

  const handleDelete = async (id: number) => {
    if (confirm('Are you sure you want to delete this protein?')) {
      await deleteMutation.mutateAsync(id)
    }
  }

  const handleVisualize = (proteinId: number) => {
    navigate(`/proteins/${proteinId}/visualize`)
  }

  if (isLoading) {
    return <div className="text-center">Loading proteins...</div>
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-black">
          Protein Library
        </h1>
        <button
          onClick={() => setShowUploadModal(true)}
          className="btn-primary"
        >
          + Add Protein
        </button>
      </div>

      {proteins && proteins.length === 0 && (
        <div className="card text-center py-12">
          <p className="text-gray-600 mb-4">
            No proteins yet. Upload your first protein to get started!
          </p>
          <button
            onClick={() => setShowUploadModal(true)}
            className="btn-primary"
          >
            Upload Protein
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 gap-4">
        {proteins?.map((protein: Protein) => (
          <div key={protein.id} className="card">
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <h3 className="text-xl font-semibold text-black mb-2">
                  {protein.name}
                </h3>
                {protein.description && (
                  <p className="text-gray-700 mb-2">
                    {protein.description}
                  </p>
                )}
                <div className="flex gap-4 text-sm text-gray-600">
                  <span>Length: {protein.sequence_length} aa</span>
                  <span>
                    Added: {new Date(protein.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => handleVisualize(protein.id)}
                  className="btn-primary text-sm"
                >
                  Visualize
                </button>
                <button
                  onClick={() => handleDelete(protein.id)}
                  className="btn-secondary text-sm"
                  disabled={deleteMutation.isPending}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {showUploadModal && (
        <ProteinUploadModal onClose={() => setShowUploadModal(false)} />
      )}
    </div>
  )
}
