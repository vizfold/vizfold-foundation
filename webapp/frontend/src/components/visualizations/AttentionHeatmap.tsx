import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { AttentionWeights } from '../../types'

interface AttentionHeatmapProps {
  attentionWeights?: AttentionWeights
  sequenceLength: number
}

export default function AttentionHeatmap({ attentionWeights, sequenceLength }: AttentionHeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 600, height: 600 })

  // Update dimensions on resize
  useEffect(() => {
    if (!containerRef.current) return
    const updateDimensions = () => {
      const containerWidth = containerRef.current?.clientWidth || 600
      setDimensions({ width: containerWidth, height: containerWidth })
    }
    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  useEffect(() => {
    if (!attentionWeights || !svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const { width, height } = dimensions
    const margin = { top: 60, right: 100, bottom: 60, left: 80 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Create matrix from attention weights
    const matrix: number[][] = Array(sequenceLength)
      .fill(0)
      .map(() => Array(sequenceLength).fill(0))

    attentionWeights.weights.forEach((weight) => {
      if (weight.source < sequenceLength && weight.target < sequenceLength) {
        matrix[weight.source][weight.target] = weight.weight
      }
    })

    const maxValue = d3.max(matrix.flat()) || 1

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, sequenceLength])
      .range([0, innerWidth])

    const yScale = d3.scaleLinear()
      .domain([0, sequenceLength])
      .range([0, innerHeight])

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, maxValue])

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([1, 10])
      .translateExtent([[0, 0], [innerWidth, innerHeight]])
      .on('zoom', (event) => {
        mainGroup.attr('transform', event.transform)
        // Update axes on zoom
        gX.call(xAxis.scale(event.transform.rescaleX(xScale)))
        gY.call(yAxis.scale(event.transform.rescaleY(yScale)))
      })

    svg
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', [0, 0, width, height])
      .call(zoom as any)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const mainGroup = g.append('g')

    // Draw heatmap cells
    const cellWidth = innerWidth / sequenceLength
    const cellHeight = innerHeight / sequenceLength

    matrix.forEach((row, i) => {
      row.forEach((value, j) => {
        if (value > 0) {
          mainGroup.append('rect')
            .attr('x', j * cellWidth)
            .attr('y', i * cellHeight)
            .attr('width', cellWidth)
            .attr('height', cellHeight)
            .attr('fill', colorScale(value))
            .attr('stroke', 'none')
            .on('mouseover', function (event) {
              d3.select(this).attr('stroke', '#fff').attr('stroke-width', 2)
              // Show tooltip
              const [mouseX, mouseY] = d3.pointer(event, svg.node())
              svg.append('g')
                .attr('class', 'tooltip')
                .attr('transform', `translate(${mouseX + 10},${mouseY - 10})`)
                .call(g => {
                  g.append('rect')
                    .attr('fill', 'rgba(0, 0, 0, 0.8)')
                    .attr('rx', 4)
                    .attr('width', 120)
                    .attr('height', 50)
                  g.append('text')
                    .attr('x', 60)
                    .attr('y', 20)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '12px')
                    .attr('fill', 'white')
                    .text(`Source: ${i}`)
                  g.append('text')
                    .attr('x', 60)
                    .attr('y', 35)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '12px')
                    .attr('fill', 'white')
                    .text(`Target: ${j}`)
                  g.append('text')
                    .attr('x', 60)
                    .attr('y', 50)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '12px')
                    .attr('font-weight', 'bold')
                    .attr('fill', '#4ade80')
                    .text(`${value.toFixed(4)}`)
                })
            })
            .on('mouseout', function () {
              d3.select(this).attr('stroke', 'none')
              svg.selectAll('.tooltip').remove()
            })
        }
      })
    })

    // Add axes
    const xAxis = d3.axisBottom(xScale)
      .ticks(10)
      .tickFormat(d => d.toString())

    const yAxis = d3.axisLeft(yScale)
      .ticks(10)
      .tickFormat(d => d.toString())

    const gX = g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(xAxis)
      .attr('color', '#888')

    const gY = g.append('g')
      .call(yAxis)
      .attr('color', '#888')

    // Add labels
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - 15)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', '#666')
      .text('Target Residue Position')

    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -height / 2)
      .attr('y', 20)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('fill', '#666')
      .text('Source Residue Position')

    // Title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', 25)
      .attr('text-anchor', 'middle')
      .attr('font-size', '16px')
      .attr('font-weight', 'bold')
      .attr('fill', '#333')
      .text(`Interactive Heatmap - Layer ${attentionWeights.layer}, Head ${attentionWeights.head}`)

    // Add color legend
    const legendWidth = 20
    const legendHeight = innerHeight
    const legendScale = d3.scaleLinear()
      .domain([0, maxValue])
      .range([legendHeight, 0])

    const legend = svg.append('g')
      .attr('transform', `translate(${width - margin.right + 30},${margin.top})`)

    // Create gradient
    const defs = svg.append('defs')
    const gradient = defs.append('linearGradient')
      .attr('id', 'heatmap-gradient')
      .attr('x1', '0%')
      .attr('x2', '0%')
      .attr('y1', '100%')
      .attr('y2', '0%')

    const steps = 10
    d3.range(steps + 1).forEach(i => {
      const value = (i / steps) * maxValue
      gradient.append('stop')
        .attr('offset', `${(i / steps) * 100}%`)
        .attr('stop-color', colorScale(value))
    })

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#heatmap-gradient)')

    const legendAxis = d3.axisRight(legendScale)
      .ticks(5)
      .tickFormat(d => d3.format('.2f')(d as number))

    legend.append('g')
      .attr('transform', `translate(${legendWidth}, 0)`)
      .call(legendAxis)
      .attr('color', '#888')

    legend.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -legendHeight / 2)
      .attr('y', legendWidth + 50)
      .attr('text-anchor', 'middle')
      .attr('font-size', '12px')
      .attr('fill', '#666')
      .text('Attention Weight')

    // Add instructions
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', height - margin.bottom + 45)
      .attr('text-anchor', 'middle')
      .attr('font-size', '11px')
      .attr('fill', '#999')
      .text('Scroll to zoom • Drag to pan • Hover for details')

  }, [attentionWeights, sequenceLength, dimensions])

  if (!attentionWeights) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-500">
        No attention data available
      </div>
    )
  }

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef}></svg>
    </div>
  )
}
