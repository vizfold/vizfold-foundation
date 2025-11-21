import { useEffect, useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Line } from '@react-three/drei'
import * as THREE from 'three'
import type { Protein, AttentionWeights } from '../../types'
import { proteinsApi } from '../../services/api'

interface Protein3DViewerProps {
  protein: Protein
  attentionWeights?: AttentionWeights
  imageUrl?: string
  onGenerate?: () => void
  isGenerating?: boolean
  interactive?: boolean
}

interface ResiduePoint {
  index: number
  residueNumber: number
  x: number
  y: number
  z: number
}

interface EdgeSegment {
  id: string
  points: [number, number, number][]
  color: string
  strength: number
}

function parsePdbCoordinates(pdbText: string): ResiduePoint[] {
  const lines = pdbText.split('\n')
  const residues: ResiduePoint[] = []
  let residueIndex = 0

  for (const line of lines) {
    if (line.startsWith('ATOM') && line.slice(12, 16).trim() === 'CA') {
      const residueNumber = parseInt(line.slice(22, 26).trim(), 10)
      const x = parseFloat(line.slice(30, 38))
      const y = parseFloat(line.slice(38, 46))
      const z = parseFloat(line.slice(46, 54))

      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        residues.push({
          index: residueIndex,
          residueNumber,
          x,
          y,
          z,
        })
        residueIndex += 1
      }
    }
  }

  return residues
}

function normalizeResidues(residues: ResiduePoint[]): ResiduePoint[] {
  if (residues.length === 0) return residues

  const xs = residues.map(r => r.x)
  const ys = residues.map(r => r.y)
  const zs = residues.map(r => r.z)

  const centerX = (Math.max(...xs) + Math.min(...xs)) / 2
  const centerY = (Math.max(...ys) + Math.min(...ys)) / 2
  const centerZ = (Math.max(...zs) + Math.min(...zs)) / 2

  const span = Math.max(
    Math.max(...xs) - Math.min(...xs),
    Math.max(...ys) - Math.min(...ys),
    Math.max(...zs) - Math.min(...zs),
    1
  )

  const scale = 30 / span

  return residues.map((r) => ({
    ...r,
    x: (r.x - centerX) * scale,
    y: (r.y - centerY) * scale,
    z: (r.z - centerZ) * scale,
  }))
}

function ProteinScene({ residues, edges }: { residues: ResiduePoint[]; edges: EdgeSegment[] }) {
  const positions = useMemo(() => residues.map(r => [r.x, r.y, r.z] as [number, number, number]), [residues])

  return (
    <>
      <ambientLight intensity={0.7} />
      <directionalLight position={[20, 20, 20]} intensity={0.5} />
      <directionalLight position={[-20, -10, -10]} intensity={0.3} />

      <Line
        points={positions}
        color="#111"
        lineWidth={1}
        opacity={0.8}
        transparent
      />

      {residues.map((residue) => (
        <mesh key={residue.index} position={[residue.x, residue.y, residue.z]}>
          <sphereGeometry args={[0.6, 16, 16]} />
          <meshStandardMaterial color="#111" roughness={0.3} metalness={0.1} />
        </mesh>
      ))}

      {edges.map((edge) => (
        <Line
          key={edge.id}
          points={edge.points}
          color={edge.color}
          lineWidth={Math.max(1, edge.strength * 6)}
          opacity={0.8}
          transparent
        />
      ))}

      <OrbitControls enableDamping />
    </>
  )
}

export default function Protein3DViewer({
  protein,
  attentionWeights,
  imageUrl,
  onGenerate,
  isGenerating = false,
  interactive = false
}: Protein3DViewerProps) {
  const [imageError, setImageError] = useState(false)
  const [imageLoaded, setImageLoaded] = useState(false)
  const [structure, setStructure] = useState<ResiduePoint[] | null>(null)
  const [structureError, setStructureError] = useState<string | null>(null)
  const [loadingStructure, setLoadingStructure] = useState(false)

  useEffect(() => {
    if (!interactive) return
    let cancelled = false

    const loadStructure = async () => {
      try {
        setLoadingStructure(true)
        setStructureError(null)
        const pdbText = await proteinsApi.getPdb(protein.id)
        if (cancelled) return
        const residues = normalizeResidues(parsePdbCoordinates(pdbText))
        if (residues.length === 0) {
          setStructureError('Could not parse CA coordinates from PDB file.')
          setStructure(null)
        } else {
          setStructure(residues)
        }
      } catch (error) {
        if (!cancelled) {
          setStructureError('PDB structure unavailable. Upload a PDB or run inference.')
          setStructure(null)
        }
      } finally {
        if (!cancelled) {
          setLoadingStructure(false)
        }
      }
    }

    loadStructure()

    return () => {
      cancelled = true
    }
  }, [interactive, protein.id])

  const attentionEdges = useMemo<EdgeSegment[]>(() => {
    if (!structure || !attentionWeights) return []

    const limited = attentionWeights.weights.slice(0, 200)
    if (limited.length === 0) return []

    const weights = limited.map(w => w.weight)
    const minWeight = Math.min(...weights)
    const maxWeight = Math.max(...weights)

    return limited
      .map((weight, idx) => {
        const sourceResidue = structure[weight.source]
        const targetResidue = structure[weight.target]
        if (!sourceResidue || !targetResidue) return null

        const normalized = maxWeight === minWeight
          ? 0.5
          : (weight.weight - minWeight) / (maxWeight - minWeight)

        const color = new THREE.Color()
        color.setHSL(0.08 + (1 - normalized) * 0.55, 0.85, 0.5)

        return {
          id: `${idx}-${weight.source}-${weight.target}`,
          points: [
            [sourceResidue.x, sourceResidue.y, sourceResidue.z],
            [targetResidue.x, targetResidue.y, targetResidue.z],
          ] as [number, number, number][],
          color: color.getStyle(),
          strength: weight.weight,
        }
      })
      .filter(Boolean) as EdgeSegment[]
  }, [attentionWeights, structure])

  const handleImageError = () => {
    setImageError(true)
    setImageLoaded(false)
  }

  const handleImageLoad = () => {
    setImageError(false)
    setImageLoaded(true)
  }

  if (interactive) {
    if (loadingStructure) {
      return (
        <div className="flex flex-col items-center justify-center h-96 text-center space-y-2">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-black" />
          <p className="text-sm text-gray-700">Loading PDB structure…</p>
        </div>
      )
    }

    if (structureError) {
      return (
        <div className="flex flex-col items-center justify-center h-96 text-center space-y-4">
          <p className="text-sm text-gray-700">{structureError}</p>
          {onGenerate && (
            <button className="btn-primary" onClick={onGenerate} disabled={isGenerating}>
              {isGenerating ? 'Generating…' : 'Generate Static Snapshot'}
            </button>
          )}
        </div>
      )
    }

    if (!structure || !attentionWeights || attentionEdges.length === 0) {
      return (
        <div className="flex flex-col items-center justify-center h-96 text-center space-y-3">
          <p className="text-sm text-gray-700">Attention weights not available for this selection.</p>
          <p className="text-xs text-gray-500">Run inference or switch to a layer/head with stored weights.</p>
        </div>
      )
    }

    return (
      <div className="space-y-3">
        <div className="w-full h-[420px] border border-gray-200 rounded-lg overflow-hidden bg-white">
          <Canvas camera={{ position: [0, 0, 45], fov: 45 }}>
            <ProteinScene residues={structure} edges={attentionEdges} />
          </Canvas>
        </div>
        <div className="text-xs text-gray-600 flex justify-between">
          <span>Residues rendered: {structure.length}</span>
          <span>Edges shown: {attentionEdges.length}</span>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-full bg-white border border-gray-200 rounded-lg overflow-hidden relative">
      {imageUrl && !imageError ? (
        <>
          {!imageLoaded && !isGenerating && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-black mx-auto mb-4"></div>
                <p className="text-gray-700">Loading 3D visualization...</p>
              </div>
            </div>
          )}

          <img
            src={imageUrl}
            alt={`3D structure of ${protein.name}`}
            className={`w-full h-full object-contain ${imageLoaded ? 'opacity-100' : 'opacity-0'} transition-opacity duration-300`}
            onError={handleImageError}
            onLoad={handleImageLoad}
          />

          {imageLoaded && attentionWeights && (
            <div className="absolute bottom-2 left-2 bg-white/90 text-black text-xs p-2 border border-gray-200 rounded">
              <p className="font-semibold">{protein.name}</p>
              <p className="text-gray-700">
                Layer {attentionWeights.layer}, Head {attentionWeights.head}
              </p>
              <p className="text-gray-600">Edges: Attention connections</p>
            </div>
          )}
        </>
      ) : (
        <div className="flex flex-col items-center justify-center h-full p-8 text-center">
          {isGenerating ? (
            <>
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-black mb-4"></div>
              <p className="text-gray-800 font-medium mb-2">
                Generating 3D visualization with PyMOL...
              </p>
              <p className="text-sm text-gray-600">
                This may take 10-30 seconds
              </p>
            </>
          ) : imageError ? (
            <>
              <svg className="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-gray-800 font-medium mb-2">
                3D visualization not yet generated
              </p>
              <p className="text-sm text-gray-600 mb-4">
                Generate a PyMOL-rendered 3D structure with attention overlays
              </p>
              {onGenerate && (
                <button
                  onClick={onGenerate}
                  className="btn-primary"
                >
                  Generate 3D View
                </button>
              )}
            </>
          ) : (
            <>
              <svg className="w-16 h-16 text-gray-400 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
              </svg>
              <p className="text-gray-800 font-medium mb-2">
                No 3D visualization available
              </p>
              <p className="text-sm text-gray-600 mb-4">
                Select layer and head, then generate
              </p>
              {onGenerate && (
                <button
                  onClick={onGenerate}
                  className="btn-primary"
                >
                  Generate 3D View
                </button>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
