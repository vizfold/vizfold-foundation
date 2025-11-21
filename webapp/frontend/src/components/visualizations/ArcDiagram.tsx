import { useEffect, useRef } from 'react'
import * as d3 from 'd3'
import type { AttentionWeights } from '../../types'

interface ArcDiagramProps {
  attentionWeights?: AttentionWeights
  sequenceLength: number
}

export default function ArcDiagram({ attentionWeights, sequenceLength }: ArcDiagramProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!attentionWeights || !svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 400
    const height = 300
    const margin = { top: 40, right: 20, bottom: 40, left: 20 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Create scale for residue positions
    const xScale = d3.scaleLinear()
      .domain([0, sequenceLength - 1])
      .range([0, innerWidth])

    const topWeights = attentionWeights.weights

    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, d3.max(topWeights, d => d.weight) || 1])

    // Draw arcs
    topWeights.forEach((weight) => {
      const x1 = xScale(weight.source)
      const x2 = xScale(weight.target)
      const y = innerHeight / 2

      const midX = (x1 + x2) / 2
      const distance = Math.abs(x2 - x1)
      const radius = distance / 2
      const arcHeight = Math.min(radius, innerHeight / 2 - 20)

      const path = d3.path()
      path.moveTo(x1, y)
      path.quadraticCurveTo(midX, y - arcHeight, x2, y)

      g.append('path')
        .attr('d', path.toString())
        .attr('stroke', colorScale(weight.weight))
        .attr('stroke-width', Math.max(1, weight.weight * 3))
        .attr('fill', 'none')
        .attr('opacity', 0.6)
        .on('mouseover', function () {
          d3.select(this)
            .attr('opacity', 1)
            .attr('stroke-width', Math.max(2, weight.weight * 5))

          // Show tooltip
          g.append('text')
            .attr('class', 'tooltip')
            .attr('x', midX)
            .attr('y', y - arcHeight - 10)
            .attr('text-anchor', 'middle')
            .attr('font-size', '11px')
            .attr('fill', 'white')
            .text(`${weight.source}→${weight.target}: ${weight.weight.toFixed(3)}`)
        })
        .on('mouseout', function () {
          d3.select(this)
            .attr('opacity', 0.6)
            .attr('stroke-width', Math.max(1, weight.weight * 3))
          g.selectAll('.tooltip').remove()
        })
    })

    // Draw sequence axis
    const axisY = innerHeight / 2 + 10

    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', axisY)
      .attr('y2', axisY)
      .attr('stroke', '#888')
      .attr('stroke-width', 2)

    // Draw residue markers
    const tickInterval = Math.ceil(sequenceLength / 20)
    for (let i = 0; i < sequenceLength; i += tickInterval) {
      const x = xScale(i)

      g.append('circle')
        .attr('cx', x)
        .attr('cy', axisY)
        .attr('r', 3)
        .attr('fill', '#888')

      g.append('text')
        .attr('x', x)
        .attr('y', axisY + 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('fill', '#888')
        .text(i)
    }

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(`Arc Diagram (Layer ${attentionWeights.layer}, Head ${attentionWeights.head})`)

    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#888')
      .text(`Top ${topWeights.length} attention connections`)

  }, [attentionWeights, sequenceLength])

  if (!attentionWeights) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-500">
        No attention data available
      </div>
    )
  }

  return <svg ref={svgRef}></svg>
}
