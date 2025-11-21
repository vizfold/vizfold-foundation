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

    const width = 900
    const height = 400
    const margin = { top: 60, right: 40, bottom: 60, left: 40 }
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

    // CRITICAL FIX: DO NOT re-sort here!
    // The parent component already applied top-k filtering
    // Use the weights array as-is
    const connections = attentionWeights.weights

    if (connections.length === 0) {
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#999')
        .text('No attention connections to display')
      return
    }

    // Color scale based on weight
    const maxWeight = d3.max(connections, d => d.weight) || 1
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
      .domain([0, maxWeight])

    // Draw arcs
    const baselineY = innerHeight * 0.7

    connections.forEach((edge) => {
      const x1 = xScale(edge.source)
      const x2 = xScale(edge.target)

      const midX = (x1 + x2) / 2
      const distance = Math.abs(x2 - x1)
      const arcHeight = Math.min(distance / 2, innerHeight * 0.5)

      // Create arc path
      const path = d3.path()
      path.moveTo(x1, baselineY)
      path.quadraticCurveTo(midX, baselineY - arcHeight, x2, baselineY)

      g.append('path')
        .attr('d', path.toString())
        .attr('stroke', colorScale(edge.weight))
        .attr('stroke-width', Math.max(0.5, edge.weight * 4))
        .attr('fill', 'none')
        .attr('opacity', 0.7)
        .style('cursor', 'pointer')
        .on('mouseover', function () {
          d3.select(this)
            .attr('opacity', 1)
            .attr('stroke-width', Math.max(2, edge.weight * 6))

          // Show tooltip
          const tooltip = g.append('g')
            .attr('class', 'tooltip')

          const tooltipBg = tooltip.append('rect')
            .attr('fill', 'rgba(0, 0, 0, 0.9)')
            .attr('rx', 4)
            .attr('stroke', '#333')
            .attr('stroke-width', 1)

          const tooltipText = tooltip.append('text')
            .attr('fill', 'white')
            .attr('font-size', '11px')
            .attr('text-anchor', 'middle')

          tooltipText.append('tspan')
            .attr('x', 0)
            .attr('y', 0)
            .attr('font-weight', 'bold')
            .text(`${edge.source} → ${edge.target}`)

          tooltipText.append('tspan')
            .attr('x', 0)
            .attr('y', 14)
            .attr('fill', '#4ade80')
            .text(`Weight: ${edge.weight.toFixed(4)}`)

          // Position tooltip
          const bbox = tooltipText.node()?.getBBox()
          if (bbox) {
            tooltipBg
              .attr('x', bbox.x - 8)
              .attr('y', bbox.y - 4)
              .attr('width', bbox.width + 16)
              .attr('height', bbox.height + 8)
          }

          tooltip.attr('transform', `translate(${midX},${baselineY - arcHeight - 20})`)
        })
        .on('mouseout', function () {
          d3.select(this)
            .attr('opacity', 0.7)
            .attr('stroke-width', Math.max(0.5, edge.weight * 4))
          g.selectAll('.tooltip').remove()
        })
    })

    // Draw sequence axis
    const axisGroup = g.append('g')
      .attr('transform', `translate(0,${baselineY})`)

    axisGroup.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', 0)
      .attr('y2', 0)
      .attr('stroke', '#333')
      .attr('stroke-width', 2)

    // Draw residue markers
    const tickInterval = Math.max(1, Math.ceil(sequenceLength / 30))
    for (let i = 0; i < sequenceLength; i += tickInterval) {
      const x = xScale(i)

      axisGroup.append('circle')
        .attr('cx', x)
        .attr('cy', 0)
        .attr('r', 2.5)
        .attr('fill', '#333')

      axisGroup.append('text')
        .attr('x', x)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '9px')
        .attr('fill', '#333')
        .text(i)
    }

    // Add axis label
    axisGroup.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 40)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#666')
      .text('Residue Position')

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', '600')
      .attr('fill', '#111')
      .text(`Arc Diagram: Layer ${attentionWeights.layer}, Head ${attentionWeights.head}`)

    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 45)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#666')
      .text(`Showing ${connections.length} attention edges`)

    // Color legend
    const legendWidth = 150
    const legendHeight = 15

    const legendGroup = svg.append('g')
      .attr('transform', `translate(${width - legendWidth - margin.right - 10},${height - 35})`)

    // Create gradient for legend
    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', 'arc-legend-gradient')
      .attr('x1', '0%')
      .attr('x2', '100%')

    const steps = 10
    d3.range(steps + 1).forEach(i => {
      const value = (i / steps) * maxWeight
      gradient.append('stop')
        .attr('offset', `${(i / steps) * 100}%`)
        .attr('stop-color', colorScale(value))
    })

    legendGroup.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'url(#arc-legend-gradient)')
      .attr('stroke', '#333')
      .attr('stroke-width', 0.5)

    legendGroup.append('text')
      .attr('x', 0)
      .attr('y', -5)
      .attr('font-size', '10px')
      .attr('fill', '#333')
      .text('0')

    legendGroup.append('text')
      .attr('x', legendWidth)
      .attr('y', -5)
      .attr('text-anchor', 'end')
      .attr('font-size', '10px')
      .attr('fill', '#333')
      .text(maxWeight.toFixed(3))

    legendGroup.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('fill', '#666')
      .text('Attention Weight')

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
