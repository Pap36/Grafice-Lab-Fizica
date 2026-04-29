import Plotly from 'plotly.js-basic-dist-min'
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import type { PlotBuildResult } from '../lib/plotlyAdapter'

export interface PlotPreviewHandle {
  downloadPng: () => Promise<void>
}

interface PlotPreviewProps {
  figure: PlotBuildResult
  exportLabels: {
    title: string
    help: string
    frame: string
    frameHelp: string
    download: string
    cancel: string
  }
}

const exportImageWidth = 1200
const exportImageHeight = 850
const minimumExportFrameSize = 80

interface ExportCanvasSize {
  width: number
  height: number
}

interface ExportFrame {
  x: number
  y: number
  width: number
  height: number
}

type ExportFrameHandle = 'move' | 'n' | 's' | 'e' | 'w' | 'nw' | 'ne' | 'sw' | 'se'

const fallbackExportCanvasSize: ExportCanvasSize = {
  width: exportImageWidth,
  height: exportImageHeight
}

function defaultExportFrameForSize(size: ExportCanvasSize): ExportFrame {
  return {
    x: 0,
    y: 0,
    width: size.width,
    height: size.height
  }
}

const plotConfig = {
  responsive: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['lasso2d', 'select2d']
}

const exportPreviewConfig = {
  ...plotConfig,
  responsive: false,
  editable: true,
  edits: {
    annotationPosition: true,
    legendPosition: true,
    titleText: false
  }
}

function buildExportLayout(layout: Record<string, any>): Record<string, any> {
  const exportLayout = JSON.parse(JSON.stringify(layout)) as Record<string, any>
  const currentTopMargin = Number(exportLayout.margin?.t ?? 92)
  const titleText = exportLayout.title?.text

  exportLayout.margin = {
    ...(exportLayout.margin ?? {}),
    t: currentTopMargin + 52
  }

  exportLayout.annotations = [...(exportLayout.annotations ?? [])]

  if (titleText) {
    exportLayout.annotations.unshift({
      xref: 'paper',
      yref: 'paper',
      x: Number(exportLayout.title?.x ?? 0.03),
      y: 1.2,
      xanchor: exportLayout.title?.xanchor ?? 'left',
      yanchor: 'bottom',
      text: titleText,
      showarrow: false,
      align: 'left',
      font: exportLayout.title?.font ?? { size: 16 },
      captureevents: true
    })
    exportLayout.title = { ...(exportLayout.title ?? {}), text: '' }
  }

  if (exportLayout.legend) {
    exportLayout.legend = {
      ...exportLayout.legend,
      y: 1.04,
      yanchor: 'top'
    }
  }

  return exportLayout
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value))
}

function framePercent(value: number, total: number): string {
  return `${(value / total) * 100}%`
}

function getExportCanvasSize(node: HTMLDivElement): ExportCanvasSize {
  return {
    width: Math.max(minimumExportFrameSize, Math.round(node.clientWidth || exportImageWidth)),
    height: Math.max(minimumExportFrameSize, Math.round(node.clientHeight || exportImageHeight))
  }
}

function clampFrameToSize(frame: ExportFrame, size: ExportCanvasSize): ExportFrame {
  const width = clamp(frame.width, minimumExportFrameSize, size.width)
  const height = clamp(frame.height, minimumExportFrameSize, size.height)
  return {
    x: Math.round(clamp(frame.x, 0, size.width - width)),
    y: Math.round(clamp(frame.y, 0, size.height - height)),
    width: Math.round(width),
    height: Math.round(height)
  }
}

function scaleFrameToSize(frame: ExportFrame, fromSize: ExportCanvasSize, toSize: ExportCanvasSize): ExportFrame {
  if (fromSize.width === toSize.width && fromSize.height === toSize.height) {
    return frame
  }

  return clampFrameToSize({
    x: Math.round((frame.x / fromSize.width) * toSize.width),
    y: Math.round((frame.y / fromSize.height) * toSize.height),
    width: Math.round((frame.width / fromSize.width) * toSize.width),
    height: Math.round((frame.height / fromSize.height) * toSize.height)
  }, toSize)
}

function loadImage(url: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Could not prepare PNG crop.'))
    image.src = url
  })
}

async function cropPngToFrame(url: string, frame: ExportFrame, size: ExportCanvasSize): Promise<string> {
  const safeFrame = clampFrameToSize(frame, size)
  if (
    safeFrame.x === 0 &&
    safeFrame.y === 0 &&
    safeFrame.width === size.width &&
    safeFrame.height === size.height
  ) {
    return url
  }

  const image = await loadImage(url)
  const scaleX = image.naturalWidth / size.width
  const scaleY = image.naturalHeight / size.height
  const sourceX = Math.round(safeFrame.x * scaleX)
  const sourceY = Math.round(safeFrame.y * scaleY)
  const sourceWidth = Math.round(safeFrame.width * scaleX)
  const sourceHeight = Math.round(safeFrame.height * scaleY)
  const canvas = document.createElement('canvas')
  canvas.width = sourceWidth
  canvas.height = sourceHeight
  const context = canvas.getContext('2d')
  if (!context) {
    return url
  }

  context.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight, 0, 0, sourceWidth, sourceHeight)
  return canvas.toDataURL('image/png')
}

async function refreshMathJax(): Promise<void> {
  const mathJax = (window as any).MathJax
  if (mathJax?.typesetPromise) {
    await mathJax.typesetPromise()
  }
}

const PlotPreview = forwardRef<PlotPreviewHandle, PlotPreviewProps>(({ figure, exportLabels }, ref) => {
  const plotNodeRef = useRef<HTMLDivElement | null>(null)
  const exportPreviewRef = useRef<HTMLDivElement | null>(null)
  const [isExportPreviewOpen, setIsExportPreviewOpen] = useState(false)
  const [exportCanvasSize, setExportCanvasSize] = useState<ExportCanvasSize>(fallbackExportCanvasSize)
  const [exportFrame, setExportFrame] = useState<ExportFrame>(defaultExportFrameForSize(fallbackExportCanvasSize))

  useEffect(() => {
    if (!plotNodeRef.current) {
      return
    }
    const plotNode = plotNodeRef.current
    Plotly.react(plotNodeRef.current, figure.data, figure.layout, plotConfig).then(refreshMathJax)

    const resizeObserver = new ResizeObserver(() => {
      Plotly.Plots.resize(plotNode)
    })
    resizeObserver.observe(plotNode)

    return () => resizeObserver.disconnect()
  }, [figure])

  useImperativeHandle(ref, () => ({
    async downloadPng() {
      if (!plotNodeRef.current) {
        return
      }
      setExportCanvasSize(fallbackExportCanvasSize)
      setExportFrame(defaultExportFrameForSize(fallbackExportCanvasSize))
      setIsExportPreviewOpen(true)
    }
  }))

  function startExportFrameDrag(handle: ExportFrameHandle, event: React.PointerEvent<HTMLDivElement>) {
    const previewNode = exportPreviewRef.current
    if (!previewNode) {
      return
    }
    event.preventDefault()
    event.stopPropagation()

    const previewRect = previewNode.getBoundingClientRect()
    const startX = event.clientX
    const startY = event.clientY
    const startFrame = { ...exportFrame }
    const startSize = exportCanvasSize

    const resizeFrame = (moveEvent: PointerEvent) => {
      const deltaX = ((moveEvent.clientX - startX) / previewRect.width) * startSize.width
      const deltaY = ((moveEvent.clientY - startY) / previewRect.height) * startSize.height
      const right = startFrame.x + startFrame.width
      const bottom = startFrame.y + startFrame.height
      let nextFrame = { ...startFrame }

      if (handle === 'move') {
        nextFrame.x = clamp(startFrame.x + deltaX, 0, startSize.width - startFrame.width)
        nextFrame.y = clamp(startFrame.y + deltaY, 0, startSize.height - startFrame.height)
      }
      if (handle.includes('e')) {
        nextFrame.width = clamp(startFrame.width + deltaX, minimumExportFrameSize, startSize.width - startFrame.x)
      }
      if (handle.includes('s')) {
        nextFrame.height = clamp(startFrame.height + deltaY, minimumExportFrameSize, startSize.height - startFrame.y)
      }
      if (handle.includes('w')) {
        nextFrame.x = clamp(startFrame.x + deltaX, 0, right - minimumExportFrameSize)
        nextFrame.width = right - nextFrame.x
      }
      if (handle.includes('n')) {
        nextFrame.y = clamp(startFrame.y + deltaY, 0, bottom - minimumExportFrameSize)
        nextFrame.height = bottom - nextFrame.y
      }

      setExportFrame(clampFrameToSize({
        x: Math.round(nextFrame.x),
        y: Math.round(nextFrame.y),
        width: Math.round(nextFrame.width),
        height: Math.round(nextFrame.height)
      }, startSize))
    }

    const stopResize = () => {
      window.removeEventListener('pointermove', resizeFrame)
      window.removeEventListener('pointerup', stopResize)
    }

    window.addEventListener('pointermove', resizeFrame)
    window.addEventListener('pointerup', stopResize)
  }

  useEffect(() => {
    if (!isExportPreviewOpen || !exportPreviewRef.current) {
      return
    }

    const previewNode = exportPreviewRef.current
    const syncExportSize = (resetFrame = false) => {
      const nextSize = getExportCanvasSize(previewNode)
      setExportCanvasSize((previousSize) => {
        setExportFrame((previousFrame) => (
          resetFrame ? defaultExportFrameForSize(nextSize) : scaleFrameToSize(previousFrame, previousSize, nextSize)
        ))
        return nextSize
      })
      return nextSize
    }
    const nextSize = syncExportSize(true)
    Plotly.react(previewNode, figure.data, buildExportLayout(figure.layout), exportPreviewConfig).then(refreshMathJax)
    const resizeObserver = new ResizeObserver(() => {
      const resizedSize = syncExportSize()
      Plotly.relayout(previewNode, {
        width: resizedSize.width,
        height: resizedSize.height
      })
    })
    resizeObserver.observe(previewNode)
    Plotly.relayout(previewNode, {
      width: nextSize.width,
      height: nextSize.height
    })

    return () => {
      resizeObserver.disconnect()
      Plotly.purge(previewNode)
    }
  }, [figure, isExportPreviewOpen])

  async function downloadPreviewPng() {
    if (!exportPreviewRef.current) {
      return
    }

    const safeCanvasSize = getExportCanvasSize(exportPreviewRef.current)
    const safeFrame = clampFrameToSize(exportFrame, safeCanvasSize)
    const rawUrl = await Plotly.toImage(exportPreviewRef.current, {
      format: 'png',
      width: safeCanvasSize.width,
      height: safeCanvasSize.height,
      scale: 2
    })
    const url = await cropPngToFrame(rawUrl, safeFrame, safeCanvasSize)
    const link = document.createElement('a')
    link.href = url
    link.download = 'physics-lab-plot.png'
    link.click()
    setIsExportPreviewOpen(false)
  }

  return (
    <>
      <div className="plot-surface" ref={plotNodeRef} />
      {isExportPreviewOpen && (
        <div className="export-preview-backdrop" role="dialog" aria-modal="true" aria-label={exportLabels.title}>
          <div className="export-preview-modal">
            <header className="export-preview-header">
              <div>
                <h2>{exportLabels.title}</h2>
                <p>{exportLabels.help}</p>
              </div>
              <div className="export-preview-actions">
                <div className="export-frame-status" aria-label={exportLabels.frame} title={exportLabels.frameHelp}>
                  {exportLabels.frame}: {exportFrame.width} x {exportFrame.height}px
                </div>
                <button className="secondary-button" type="button" onClick={() => setIsExportPreviewOpen(false)}>{exportLabels.cancel}</button>
                <button className="primary-button" type="button" onClick={downloadPreviewPng}>{exportLabels.download}</button>
              </div>
            </header>
            <div className="export-preview-stage">
              <div className="export-preview-surface" ref={exportPreviewRef} />
              <div className="export-frame-overlay">
                <div className="export-frame-mask top" style={{ height: framePercent(exportFrame.y, exportCanvasSize.height) }} />
                <div
                  className="export-frame-mask bottom"
                  style={{ top: framePercent(exportFrame.y + exportFrame.height, exportCanvasSize.height) }}
                />
                <div
                  className="export-frame-mask left"
                  style={{
                    top: framePercent(exportFrame.y, exportCanvasSize.height),
                    width: framePercent(exportFrame.x, exportCanvasSize.width),
                    height: framePercent(exportFrame.height, exportCanvasSize.height)
                  }}
                />
                <div
                  className="export-frame-mask right"
                  style={{
                    top: framePercent(exportFrame.y, exportCanvasSize.height),
                    left: framePercent(exportFrame.x + exportFrame.width, exportCanvasSize.width),
                    height: framePercent(exportFrame.height, exportCanvasSize.height)
                  }}
                />
                <div
                  className="export-frame"
                  style={{
                    left: framePercent(exportFrame.x, exportCanvasSize.width),
                    top: framePercent(exportFrame.y, exportCanvasSize.height),
                    width: framePercent(exportFrame.width, exportCanvasSize.width),
                    height: framePercent(exportFrame.height, exportCanvasSize.height)
                  }}
                  title={exportLabels.frameHelp}
                >
                  <div
                    className="export-frame-grip"
                    onPointerDown={(event) => startExportFrameDrag('move', event)}
                    aria-label={exportLabels.frameHelp}
                  />
                  {(['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'] as const).map((handle) => (
                    <div
                      className={`export-frame-handle ${handle}`}
                      key={handle}
                      onPointerDown={(event) => startExportFrameDrag(handle, event)}
                    />
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
})

PlotPreview.displayName = 'PlotPreview'

export default PlotPreview
