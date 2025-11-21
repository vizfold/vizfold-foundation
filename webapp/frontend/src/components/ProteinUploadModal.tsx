import { useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useDropzone } from 'react-dropzone'
import { proteinsApi } from '../services/api'

interface ProteinUploadModalProps {
  onClose: () => void
}

export default function ProteinUploadModal({ onClose }: ProteinUploadModalProps) {
  const [mode, setMode] = useState<'sequence' | 'file'>('sequence')
  const [name, setName] = useState('')
  const [sequence, setSequence] = useState('')
  const [description, setDescription] = useState('')
  const [file, setFile] = useState<File | null>(null)

  const queryClient = useQueryClient()

  const createMutation = useMutation({
    mutationFn: proteinsApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['proteins'] })
      onClose()
    },
  })

  const uploadMutation = useMutation({
    mutationFn: ({ file, name, description }: { file: File; name: string; description?: string }) =>
      proteinsApi.upload(file, name, description),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['proteins'] })
      onClose()
    },
  })

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/plain': ['.fasta', '.fa'],
      'chemical/x-pdb': ['.pdb'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0])
      }
    },
  })

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (mode === 'sequence') {
      await createMutation.mutateAsync({ name, sequence, description })
    } else if (file) {
      await uploadMutation.mutateAsync({ file, name, description })
    }
  }

  const isLoading = createMutation.isPending || uploadMutation.isPending
  const error = createMutation.error || uploadMutation.error

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4">
        <h2 className="text-2xl font-bold mb-4 text-black">
          Add Protein
        </h2>

        <div className="flex gap-2 mb-4">
          <button
            className={`px-4 py-2 rounded border text-sm font-medium ${
              mode === 'sequence'
                ? 'bg-black text-white border-black'
                : 'bg-white text-black border-gray-300 hover:border-black'
            }`}
            onClick={() => setMode('sequence')}
          >
            Enter Sequence
          </button>
          <button
            className={`px-4 py-2 rounded border text-sm font-medium ${
              mode === 'file'
                ? 'bg-black text-white border-black'
                : 'bg-white text-black border-gray-300 hover:border-black'
            }`}
            onClick={() => setMode('file')}
          >
            Upload File
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-800">
              Protein Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="input-field"
              required
            />
          </div>

          {mode === 'sequence' ? (
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-800">
                Sequence
              </label>
              <textarea
                value={sequence}
                onChange={(e) => setSequence(e.target.value)}
                className="input-field font-mono text-sm"
                rows={8}
                placeholder="MKFLKFSLLTAVLLSVVFAFSSCGDDDDTYPYDVPDYAIEAGFPFY..."
                required
              />
              <p className="text-sm text-gray-600 mt-1">
                Enter amino acid sequence (single letter code)
              </p>
            </div>
          ) : (
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-800">
                File
              </label>
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-black bg-gray-100'
                    : 'border-gray-300 hover:border-black'
                }`}
              >
                <input {...getInputProps()} />
                {file ? (
                  <p className="text-gray-800">
                    Selected: {file.name}
                  </p>
                ) : (
                  <p className="text-gray-600">
                    {isDragActive
                      ? 'Drop file here...'
                      : 'Drag & drop a FASTA or PDB file, or click to select'}
                  </p>
                )}
              </div>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium mb-1 text-gray-800">
              Description (optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="input-field"
              rows={3}
            />
          </div>

          {error && (
            <div className="p-3 border border-gray-300 text-black rounded">
              {error instanceof Error ? error.message : 'An error occurred'}
            </div>
          )}

          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="btn-secondary"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn-primary"
              disabled={isLoading}
            >
              {isLoading ? 'Adding...' : 'Add Protein'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
