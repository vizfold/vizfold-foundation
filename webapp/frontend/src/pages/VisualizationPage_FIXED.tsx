import { useState, useEffect, useMemo } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { proteinsApi, attentionApi, visualizationsApi } from '../services/api'
import AttentionHeatmap from '../components/visualizations/AttentionHeatmap'
import ArcDiagram from '../components/visualizations/ArcDiagram'
import Protein3DViewer from '../components/visualizations/Protein3DViewer'
import StaticVisualization from '../components/visualizations/StaticVisualization'
import type { AttentionWeights } from '../types'

export default function VisualizationPage() {
  const { proteinId } = useParams<{ proteinId: string }>()
  const proteinIdNum = Number(proteinId)

  const [selectedLayer, setSelectedLayer] = useState<number>(0)
  const [selectedHead, setSelectedHead] = useState<number>(0)
  const [attentionType, setAttentionType] = useState<'msa_row' | 'triangle_start'>('msa_row')
  const [viewMode, setViewMode] = useState<'heatmap' | 'arc' | '3d' | 'combined'>('combined')
  const [useStaticViz, setUseStaticViz] = useState<boolean>(false) // Default to interactive for debugging

  // Top-k parameter - controls how many attention edges to show
  const [topK, setTopK] = useState<number>(50)

  const { data: protein } = useQuery({
    queryKey: ['protein', proteinIdNum],
    queryFn: () => proteinsApi.get(proteinIdNum),
  })

  const { data: layers } = useQuery({
    queryKey: ['attention-layers', proteinIdNum],
    queryFn: () => attentionApi.getAvailableLayers(proteinIdNum),
  })

  const { data: heads } = useQuery({
    queryKey: ['attention-heads', proteinIdNum, selectedLayer, attentionType],
    queryFn: () => attentionApi.getAvailableHeads(proteinIdNum, selectedLayer, attentionType),
    enabled: selectedLayer !== null,
  })

  const { data: attentionWeights } = useQuery({
    queryKey: ['attention-weights', proteinIdNum, selectedLayer, selectedHead, attentionType],
    queryFn: () => attentionApi.getWeights(proteinIdNum, selectedLayer, selectedHead, attentionType),
    enabled: selectedLayer !== null && selectedHead !== null && !useStaticViz,
  })

  // CRITICAL FIX: Apply consistent top-k filtering ONCE across all views
  const filteredWeights = useMemo<AttentionWeights | undefined>(() => {
    if (!attentionWeights) return undefined

    // Sort all edges by weight and take global top-k
    const sorted = [...attentionWeights.weights]
      .sort((a, b) => b.weight - a.weight)
      .slice(0, topK)

    return {
      ...attentionWeights,
      weights: sorted
    }
  }, [attentionWeights, topK])

  // Generate static visualizations using real matplotlib utilities
  const generateArcDiagram = useMutation({
    mutationFn: () => visualizationsApi.generate({
      protein_id: proteinIdNum,
      viz_type: 'arc_diagram',
      layer: selectedLayer,
      head: selectedHead,
      attention_type: attentionType,
      top_k: topK
    })
  })

  const generateHeatmap = useMutation({
    mutationFn: () => visualizationsApi.generate({
      protein_id: proteinIdNum,
      viz_type: 'heatmap',
      layer: selectedLayer,
      head: selectedHead,  // CRITICAL: Now passing head to generate L×L matrix
      attention_type: attentionType,
      top_k: topK
    })
  })

  const generate3D = useMutation({
    mutationFn: () => visualizationsApi.generate({
      protein_id: proteinIdNum,
      viz_type: '3d',
      layer: selectedLayer,
      head: selectedHead,
      attention_type: attentionType,
      top_k: topK
    })
  })

  const staticBase = `http://localhost:8000/outputs/protein_${proteinIdNum}/visualizations`
  const heatmapUrl = useStaticViz
    ? `${staticBase}/${attentionType}/attention_heatmap_layer_${selectedLayer}_protein_${proteinIdNum}.png`
    : undefined
  const arcDiagramUrl = useStaticViz
    ? `${staticBase}/arc_diagram_layer_${selectedLayer}_head_${selectedHead}.png`
    : undefined
  const view3DUrl = useStaticViz
    ? `${staticBase}/msa_row_head_${selectedHead}_layer_${selectedLayer}_protein_${proteinIdNum}.png`
    : undefined

  // Set default layer and head when data loads
  useEffect(() => {
    if (layers && layers.length > 0) {
      setSelectedLayer((prev) => (layers.includes(prev) ? prev : layers[0]))
    }
  }, [layers])

  useEffect(() => {
    if (heads && heads.length > 0) {
      setSelectedHead((prev) => (heads.includes(prev) ? prev : heads[0]))
    }
  }, [heads])

  if (!protein) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto mb-4"></div>
          <p>Loading protein...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-semibold text-black">
            {protein.name}
          </h1>
          <p className="text-sm text-gray-700 mt-1">
            {protein.sequence_length} residues
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Rendering Mode
            </label>
            <select
              value={useStaticViz ? 'static' : 'interactive'}
              onChange={(e) => setUseStaticViz(e.target.value === 'static')}
              className="input-field w-full"
            >
              <option value="interactive">Interactive (D3.js SVG)</option>
              <option value="static">Static (Accurate PNG)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">
              View Mode
            </label>
            <select
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value as any)}
              className="input-field w-full"
            >
              <option value="combined">Combined View</option>
              <option value="heatmap">Heatmap Only</option>
              <option value="arc">Arc Diagram Only</option>
              <option value="3d">3D View Only</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Attention Type
            </label>
            <select
              value={attentionType}
              onChange={(e) => setAttentionType(e.target.value as any)}
              className="input-field w-full"
            >
              <option value="msa_row">MSA Row</option>
              <option value="triangle_start">Triangle Start</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Layer
            </label>
            <select
              value={selectedLayer}
              onChange={(e) => setSelectedLayer(Number(e.target.value))}
              className="input-field w-full"
            >
              {layers?.map((layer) => (
                <option key={layer} value={layer}>
                  {layer}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Head
            </label>
            <select
              value={selectedHead}
              onChange={(e) => setSelectedHead(Number(e.target.value))}
              className="input-field w-full"
            >
              {heads?.map((head) => (
                <option key={head} value={head}>
                  {head}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Top-K Edges
            </label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="input-field w-full"
            >
              <option value={20}>20</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
              <option value={200}>200</option>
              <option value={500}>500</option>
            </select>
          </div>
        </div>

        {/* Debug info - only show in interactive mode */}
        {!useStaticViz && import.meta.env.DEV && (
          <details className="mt-4 text-xs text-gray-600">
            <summary className="cursor-pointer hover:text-gray-900">
              Debug Info (Dev Only)
            </summary>
            <div className="mt-2 p-3 bg-gray-50 rounded font-mono text-xs">
              <p>Total edges loaded: {attentionWeights?.weights.length || 0}</p>
              <p>Top-{topK} edges shown: {filteredWeights?.weights.length || 0}</p>
              <p>Sequence length: {protein.sequence_length}</p>
              {filteredWeights && filteredWeights.weights.length > 0 && (
                <div className="mt-2">
                  <p className="font-semibold">Top 5 edges:</p>
                  <ul className="list-disc list-inside">
                    {filteredWeights.weights.slice(0, 5).map((w, i) => (
                      <li key={i}>
                        {w.source} → {w.target}: {w.weight.toFixed(4)}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </details>
        )}
      </div>

      {/* Visualizations */}
      {viewMode === 'combined' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Heatmap */}
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">
              Heatmap
            </h3>
            {useStaticViz ? (
              <StaticVisualization
                imageUrl={heatmapUrl ?? ''}
                title="Attention Heatmap"
                altText={`Heatmap for layer ${selectedLayer}, head ${selectedHead}`}
                onGenerate={() => generateHeatmap.mutate()}
                isGenerating={generateHeatmap.isPending}
              />
            ) : (
              <AttentionHeatmap
                attentionWeights={filteredWeights}
                sequenceLength={protein.sequence_length}
              />
            )}
          </div>

          {/* Arc Diagram */}
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">
              Arc Diagram
            </h3>
            {useStaticViz ? (
              <StaticVisualization
                imageUrl={arcDiagramUrl ?? ''}
                title="Arc Diagram"
                altText={`Arc diagram for layer ${selectedLayer}, head ${selectedHead}`}
                onGenerate={() => generateArcDiagram.mutate()}
                isGenerating={generateArcDiagram.isPending}
              />
            ) : (
              <ArcDiagram
                attentionWeights={filteredWeights}
                sequenceLength={protein.sequence_length}
              />
            )}
          </div>

          {/* 3D Structure */}
          <div className="card">
            <h3 className="text-lg font-semibold mb-4">
              3D Structure
            </h3>
            <Protein3DViewer
              protein={protein}
              attentionWeights={filteredWeights}
              imageUrl={view3DUrl}
              onGenerate={() => generate3D.mutate()}
              isGenerating={generate3D.isPending}
              interactive={!useStaticViz}
            />
          </div>
        </div>
      )}

      {viewMode === 'heatmap' && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Attention Heatmap</h3>
          {useStaticViz ? (
            <StaticVisualization
              imageUrl={heatmapUrl ?? ''}
              title="Attention Heatmap"
              altText={`Heatmap for layer ${selectedLayer}, head ${selectedHead}`}
              onGenerate={() => generateHeatmap.mutate()}
              isGenerating={generateHeatmap.isPending}
            />
          ) : (
            <AttentionHeatmap
              attentionWeights={filteredWeights}
              sequenceLength={protein.sequence_length}
            />
          )}
        </div>
      )}

      {viewMode === 'arc' && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Arc Diagram</h3>
          {useStaticViz ? (
            <StaticVisualization
              imageUrl={arcDiagramUrl ?? ''}
              title="Arc Diagram"
              altText={`Arc diagram for layer ${selectedLayer}, head ${selectedHead}`}
              onGenerate={() => generateArcDiagram.mutate()}
              isGenerating={generateArcDiagram.isPending}
            />
          ) : (
            <ArcDiagram
              attentionWeights={filteredWeights}
              sequenceLength={protein.sequence_length}
            />
          )}
        </div>
      )}

      {viewMode === '3d' && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">3D Structure</h3>
          <Protein3DViewer
            protein={protein}
            attentionWeights={filteredWeights}
            imageUrl={view3DUrl}
            onGenerate={() => generate3D.mutate()}
            isGenerating={generate3D.isPending}
            interactive={!useStaticViz}
          />
        </div>
      )}
    </div>
  )
}
