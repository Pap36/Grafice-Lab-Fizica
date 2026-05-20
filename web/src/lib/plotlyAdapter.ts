import type { EditablePlotConfig, PlotItemConfig, SeriesConfig, WorkbookPlotConfig, WorksheetData } from '../types/templates'
import { getTranslations, type AppLanguage } from './i18n'
import {
  axisLabelWithUnit,
  buildFormulaCurve,
  exponentialFit,
  exponentialValue,
  extractNumericSeries,
  formatLatexText,
  formatMathAnnotation,
  formatPlotTitle,
  gaussianFit,
  gaussianValue,
  linearFit,
  linspace,
  polynomialFit,
  polynomialValue,
  scaledTickText
} from './plotMath'

export interface PlotBuildResult {
  data: any[]
  layout: any
  summaries: string[]
  errors: string[]
  recomputeTicks: (xRange: [number, number], yRange: [number, number]) => Record<string, unknown>
}

const palette = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#be123c']

function dashStyle(linestyle?: string): string {
  switch (linestyle) {
    case '--':
      return 'dash'
    case ':':
      return 'dot'
    case '-.':
      return 'dashdot'
    default:
      return 'solid'
  }
}

function markerSymbol(marker?: string | null): string | undefined {
  switch (marker) {
    case 's':
      return 'square'
    case '^':
      return 'triangle-up'
    case 'v':
      return 'triangle-down'
    case 'x':
      return 'x'
    case null:
      return undefined
    default:
      return 'circle'
  }
}

function stepShape(config: WorkbookPlotConfig): string {
  switch (config.step_where) {
    case 'pre':
      return 'vh'
    case 'post':
      return 'hv'
    default:
      return 'hvh'
  }
}

function traceColor(config: SeriesConfig, index: number): string {
  return config.color || palette[index % palette.length]
}

function traceName(name: string | undefined): string | undefined {
  return name ? formatLatexText(name) : name
}

function lineTrace(
  name: string | undefined,
  xValues: number[],
  yValues: number[],
  config: SeriesConfig,
  index: number,
  mode: string,
  shape?: string
): any {
  const symbol = markerSymbol(config.marker)
  return {
    type: 'scatter',
    mode: symbol ? mode : mode.replace('+markers', ''),
    name: traceName(name),
    x: xValues,
    y: yValues,
    line: {
      color: traceColor(config, index),
      width: Number(config.linewidth ?? 2),
      dash: dashStyle(config.linestyle),
      shape
    },
    marker: {
      color: traceColor(config, index),
      size: 8,
      symbol
    },
    hovertemplate: 'x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>'
  }
}

function scatterTrace(name: string | undefined, xValues: number[], yValues: number[], config: SeriesConfig, index: number): any {
  return {
    type: 'scatter',
    mode: 'markers',
    name: traceName(name),
    x: xValues,
    y: yValues,
    marker: {
      color: traceColor(config, index),
      size: 8,
      symbol: markerSymbol(config.marker) || 'circle'
    },
    hovertemplate: 'x=%{x}<br>y=%{y}<extra>%{fullData.name}</extra>'
  }
}

function legendTextOnlyTrace(name: string, id: string, rank: number): any {
  return {
    type: 'scatter',
    mode: 'lines',
    name: traceName(name),
    x: [null],
    y: [null],
    showlegend: true,
    hoverinfo: 'skip',
    legendgroup: id,
    legendrank: rank,
    line: {
      color: 'rgba(0,0,0,0)',
      width: 0
    },
    marker: {
      color: 'rgba(0,0,0,0)',
      size: 0,
      opacity: 0
    }
  }
}

function annotationPosition(position?: string): Pick<any, 'x' | 'y' | 'xanchor' | 'yanchor'> {
  switch (position) {
    case 'top-left':
      return { x: 0.02, y: 0.98, xanchor: 'left', yanchor: 'top' }
    case 'bottom-left':
      return { x: 0.02, y: 0.02, xanchor: 'left', yanchor: 'bottom' }
    case 'bottom-right':
      return { x: 0.98, y: 0.02, xanchor: 'right', yanchor: 'bottom' }
    default:
      return { x: 0.98, y: 0.98, xanchor: 'right', yanchor: 'top' }
  }
}

function niceStep(rawStep: number): number {
  if (rawStep <= 0 || !Number.isFinite(rawStep)) return 1
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)))
  const normalized = rawStep / magnitude
  if (normalized <= 1) return magnitude
  if (normalized <= 2) return 2 * magnitude
  if (normalized <= 5) return 5 * magnitude
  return 10 * magnitude
}

function niceTickValues(min: number, max: number, targetCount = 6, fixedStep?: number | null): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max) || min >= max) return [min]
  const useFixedStep = fixedStep !== undefined && fixedStep !== null && Number.isFinite(fixedStep) && fixedStep > 0
  const step = useFixedStep ? Number(fixedStep) : niceStep((max - min) / (targetCount - 1))
  const start = parseFloat((Math.floor(min / step) * step).toPrecision(12))
  const end = parseFloat((Math.ceil(max / step) * step).toPrecision(12))
  const ticks: number[] = []
  let current = start
  const safetyLimit = useFixedStep ? 1000 : 200
  while (current <= end + step * 1e-9 && ticks.length < safetyLimit) {
    ticks.push(parseFloat(current.toPrecision(12)))
    current += step
  }
  return ticks.length > 0 ? ticks : [min]
}

export interface AxisFormatting {
  step?: number | null
  precision?: number | null
}

export function computeAxisTickUpdates(
  xRange: [number, number],
  yRange: [number, number],
  xExponent: number,
  yExponent: number,
  xFormatting: AxisFormatting = {},
  yFormatting: AxisFormatting = {}
): Record<string, unknown> {
  const xTickValues = niceTickValues(xRange[0], xRange[1], 6, xFormatting.step)
  const yTickValues = niceTickValues(yRange[0], yRange[1], 6, yFormatting.step)
  const xTickText = xTickValues.map((value) => scaledTickText(value, xExponent, xFormatting.precision))
  const yTickText = yTickValues.map((value) => scaledTickText(value, yExponent, yFormatting.precision))
  if (xTickText.length > 1 && yTickValues.length > 0) {
    xTickText[0] = ''
  }
  return {
    'xaxis.tickvals': xTickValues,
    'xaxis.ticktext': xTickText,
    'yaxis.tickvals': yTickValues,
    'yaxis.ticktext': yTickText
  }
}

function buildTicks(
  values: number[],
  exponent: number,
  forceMin?: number,
  forceMax?: number,
  formatting: AxisFormatting = {}
): { tickvals?: number[]; ticktext?: string[]; range?: [number, number] } {
  const finiteValues = values.filter(Number.isFinite)
  if (finiteValues.length === 0) return {}
  const dataMin = Math.min(...finiteValues)
  const dataMax = Math.max(...finiteValues)
  const effectiveMin = forceMin !== undefined ? forceMin : dataMin
  const effectiveMax = forceMax !== undefined ? forceMax : dataMax
  if (effectiveMin === effectiveMax) {
    return {
      tickvals: [effectiveMin],
      ticktext: [scaledTickText(effectiveMin, exponent, formatting.precision)],
      range: [effectiveMin - 1, effectiveMin + 1]
    }
  }
  const tickValues = niceTickValues(effectiveMin, effectiveMax, 6, formatting.step)
  return {
    tickvals: tickValues,
    ticktext: tickValues.map((value) => scaledTickText(value, exponent, formatting.precision)),
    range: [tickValues[0], tickValues[tickValues.length - 1]]
  }
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

function statLine(label: string, value: number, precision = 5): string {
  return `${escapeHtml(formatLatexText(label))} = ${value.toFixed(precision)}`
}

function rSquaredLine(rSquared: number): string {
  return `R² = ${rSquared.toFixed(5)}`
}

function linearStatsLines(config: EditablePlotConfig, plot: WorkbookPlotConfig, name: string, slope: number, intercept: number, rSquared: number): string[] {
  if (plot.show_stats === false) {
    return []
  }
  const lines: string[] = []
  if (plot.show_slope ?? config.show_slope ?? true) {
    const exponent = Number(config.slope_exponent ?? 1)
    const display = exponent !== 0 ? slope / 10 ** exponent : slope
    const exponentText = exponent !== 0 ? ` · 10${formatLatexText(`^{${exponent}}`)}` : ''
    const unit = config.slope_unit ? ` (${formatLatexText(config.slope_unit)})` : ''
    lines.push(`${escapeHtml(formatLatexText(config.slope_label || 'm'))} = ${display.toFixed(Number(config.slope_precision ?? 5))}${exponentText}${unit}`)
  }
  if (plot.show_intercept ?? config.show_intercept ?? true) {
    const exponent = Number(config.intercept_exponent ?? 1)
    const display = exponent !== 0 ? intercept / 10 ** exponent : intercept
    const exponentText = exponent !== 0 ? ` · 10${formatLatexText(`^{${exponent}}`)}` : ''
    const unit = config.intercept_unit ? ` (${formatLatexText(config.intercept_unit)})` : ''
    lines.push(`${escapeHtml(formatLatexText(config.intercept_label || 'b'))} = ${display.toFixed(Number(config.intercept_precision ?? 5))}${exponentText}${unit}`)
  }
  if (plot.show_r_squared ?? true) {
    lines.push(rSquaredLine(rSquared))
  }
  return lines.length ? [`<b>${escapeHtml(name)}</b>`, ...lines] : []
}

function polynomialStatsLines(plot: WorkbookPlotConfig, name: string, coefficients: number[], rSquared: number): string[] {
  if (plot.show_stats === false) {
    return []
  }
  const lines: string[] = []
  if (plot.show_polynomial_coefficients ?? true) {
    coefficients.forEach((coefficient, index) => {
      lines.push(statLine(`a_${index}`, coefficient))
    })
  }
  if (plot.show_r_squared ?? true) {
    lines.push(rSquaredLine(rSquared))
  }
  return lines.length ? [`<b>${escapeHtml(name)}</b>`, ...lines] : []
}

function exponentialStatsLines(plot: WorkbookPlotConfig, name: string, coefficient: number, rate: number, rSquared: number): string[] {
  if (plot.show_stats === false) {
    return []
  }
  const lines: string[] = []
  if (plot.show_exponential_coefficient ?? true) {
    lines.push(statLine('A', coefficient))
  }
  if (plot.show_exponential_rate ?? true) {
    lines.push(statLine('b', rate))
  }
  if (plot.show_r_squared ?? true) {
    lines.push(rSquaredLine(rSquared))
  }
  return lines.length ? [`<b>${escapeHtml(name)}</b>`, ...lines] : []
}

function gaussianStatsLines(plot: WorkbookPlotConfig, name: string, fit: ReturnType<typeof gaussianFit>): string[] {
  if (plot.show_stats === false) {
    return []
  }
  const lines: string[] = []
  if (plot.show_gaussian_baseline ?? true) {
    lines.push(statLine('c', fit.baseline))
  }
  if (plot.show_gaussian_amplitude ?? true) {
    lines.push(statLine('A', fit.amplitude))
  }
  if (plot.show_gaussian_mean ?? true) {
    lines.push(statLine('\\mu', fit.mean))
  }
  if (plot.show_gaussian_std ?? true) {
    lines.push(statLine('\\sigma', fit.sigma))
  }
  if (plot.show_r_squared ?? true) {
    lines.push(rSquaredLine(fit.rSquared))
  }
  return lines.length ? [`<b>${escapeHtml(name)}</b>`, ...lines] : []
}

function polynomialEquation(coefficients: number[]): string {
  return coefficients
    .map((coefficient, index) => {
      if (index === 0) {
        return coefficient.toFixed(6)
      }
      if (index === 1) {
        return `${coefficient.toFixed(6)} * x`
      }
      return `${coefficient.toFixed(6)} * x^${index}`
    })
    .join(' + ')
    .replace(/\+ -/g, '- ')
}

function fitDomain(plot: WorkbookPlotConfig, xValues: number[], globalMinimum: number, globalMaximum: number): number[] {
  const localMinimum = plot.start_at_zero ? 0 : Math.min(...xValues)
  const localMaximum = Math.max(...xValues)
  const minimum = plot.extrapolate ? Math.min(globalMinimum, localMinimum) : localMinimum
  const maximum = plot.extrapolate ? Math.max(globalMaximum, localMaximum) : localMaximum
  return linspace(minimum, maximum, 240)
}

export function buildPlotFigure(config: EditablePlotConfig, sheet: WorksheetData | null, language: AppLanguage = 'en'): PlotBuildResult {
  const t = getTranslations(language)
  const errors: string[] = []
  const summaries: string[] = []
  const data: any[] = []
  const allXValues: number[] = []
  const allYValues: number[] = []
  const annotationLines: string[] = []

  const xExponent = Number(config.x_exponent ?? 1)
  const yExponent = Number(config.y_exponent ?? 1)
  const xFormatting: AxisFormatting = { step: config.x_tick_step, precision: config.x_tick_precision }
  const yFormatting: AxisFormatting = { step: config.y_tick_step, precision: config.y_tick_precision }
  const recomputeTicks = (xRange: [number, number], yRange: [number, number]) =>
    computeAxisTickUpdates(xRange, yRange, xExponent, yExponent, xFormatting, yFormatting)

  const plots = config.plots ?? []
  if (plots.length === 0) {
    errors.push(t.addPlotToBegin)
    return { data, summaries, errors, layout: baseLayout(config, [], []), recomputeTicks }
  }

  const preparedWorkbookPlots: Array<{ plot: WorkbookPlotConfig; xValues: number[]; yValues: number[]; index: number }> = []
  const formulaPlots: PlotItemConfig[] = []

  plots.forEach((plot, index) => {
    if (plot.source_type === 'formula') {
      formulaPlots.push(plot)
      return
    }
    if (!sheet) {
      errors.push(`${plot.label || t.plot(index + 1)}: ${t.uploadFileAndChooseSheet}`)
      return
    }
    try {
      const series = extractNumericSeries(sheet, plot)
      preparedWorkbookPlots.push({ plot, xValues: series.xValues, yValues: series.yValues, index })
      allXValues.push(...series.xValues, ...(plot.start_at_zero ? [0] : []))
      allYValues.push(...series.yValues)
    } catch (error) {
      errors.push(`${plot.label || `Plot ${index + 1}`}: ${error instanceof Error ? error.message : String(error)}`)
    }
  })

  const formulaCurves = formulaPlots.flatMap((plot, index) => {
    try {
      const curve = buildFormulaCurve(
        plot,
        allXValues.length ? Math.min(...allXValues) : null,
        allXValues.length ? Math.max(...allXValues) : null
      )
      allXValues.push(...curve.xValues)
      allYValues.push(...curve.yValues)
      return [{ curve, index: index + preparedWorkbookPlots.length }]
    } catch (error) {
      errors.push(`${plot.label || `Formula ${index + 1}`}: ${error instanceof Error ? error.message : String(error)}`)
      return []
    }
  })

  const globalMinimum = allXValues.length ? Math.min(...allXValues, config.x_start_at_zero ? 0 : Number.POSITIVE_INFINITY) : 0
  const globalMaximum = allXValues.length ? Math.max(...allXValues) : 1

  preparedWorkbookPlots.forEach(({ plot, xValues, yValues, index }) => {
    const rawName = plot.label || `Plot ${index + 1}`
    const name = formatLatexText(rawName)
    const color = traceColor(plot, index)
    const legendGroup = plot.id || `plot-${index}`

    if (plot.render_mode === 'fit') {
      if (plot.show_points ?? true) {
        data.push({ ...scatterTrace(name, xValues, yValues, plot, index), legendgroup: legendGroup, showlegend: true })
      }
      const lineXValues = fitDomain(plot, xValues, globalMinimum, globalMaximum)
      const fitModel = plot.fit_model ?? 'linear'
      const fitTraceBase = { name, legendgroup: legendGroup, showlegend: !(plot.show_points ?? true) }
      if (fitModel === 'gaussian') {
        try {
          const fit = gaussianFit(xValues, yValues)
          const lineYValues = lineXValues.map((value) => gaussianValue(value, fit))
          data.push({
            ...fitTraceBase,
            type: 'scatter',
            mode: 'lines',
            x: lineXValues,
            y: lineYValues,
            line: { color, width: Number(plot.linewidth ?? 2), dash: dashStyle(plot.linestyle ?? '--') },
            hovertemplate: 'x=%{x}<br>fit=%{y}<extra>%{fullData.name}</extra>'
          })
          summaries.push(`${name}: y = ${fit.baseline.toFixed(6)} + ${fit.amplitude.toFixed(6)} * exp(-((x - ${fit.mean.toFixed(6)})^2) / (2 * ${fit.sigma.toFixed(6)}^2)) (R^2 = ${fit.rSquared.toFixed(6)})`)
          annotationLines.push(...gaussianStatsLines(plot, name, fit))
        } catch (error) {
          errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
        }
      } else if (fitModel === 'polynomial') {
        try {
          const fit = polynomialFit(xValues, yValues, plot.polynomial_degree ?? 2)
          const lineYValues = lineXValues.map((value) => polynomialValue(value, fit.coefficients))
          data.push({
            ...fitTraceBase,
            type: 'scatter',
            mode: 'lines',
            x: lineXValues,
            y: lineYValues,
            line: { color, width: Number(plot.linewidth ?? 2), dash: dashStyle(plot.linestyle ?? '--') },
            hovertemplate: 'x=%{x}<br>fit=%{y}<extra>%{fullData.name}</extra>'
          })
          summaries.push(`${name}: y = ${polynomialEquation(fit.coefficients)} (R^2 = ${fit.rSquared.toFixed(6)})`)
          annotationLines.push(...polynomialStatsLines(plot, name, fit.coefficients, fit.rSquared))
        } catch (error) {
          errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
        }
      } else if (fitModel === 'exponential') {
        try {
          const fit = exponentialFit(xValues, yValues)
          const lineYValues = lineXValues.map((value) => exponentialValue(value, fit))
          data.push({
            ...fitTraceBase,
            type: 'scatter',
            mode: 'lines',
            x: lineXValues,
            y: lineYValues,
            line: { color, width: Number(plot.linewidth ?? 2), dash: dashStyle(plot.linestyle ?? '--') },
            hovertemplate: 'x=%{x}<br>fit=%{y}<extra>%{fullData.name}</extra>'
          })
          summaries.push(`${name}: y = ${fit.coefficient.toFixed(6)} * exp(${fit.rate.toFixed(6)} * x) (R^2 = ${fit.rSquared.toFixed(6)})`)
          annotationLines.push(...exponentialStatsLines(plot, name, fit.coefficient, fit.rate, fit.rSquared))
        } catch (error) {
          errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
        }
      } else {
        try {
          const fit = linearFit(xValues, yValues, Boolean(plot.force_through_origin))
          const lineYValues = lineXValues.map((value) => fit.slope * value + fit.intercept)
          data.push({
            ...fitTraceBase,
            type: 'scatter',
            mode: 'lines',
            x: lineXValues,
            y: lineYValues,
            line: { color, width: Number(plot.linewidth ?? 2), dash: dashStyle(plot.linestyle ?? '--') },
            hovertemplate: 'x=%{x}<br>fit=%{y}<extra>%{fullData.name}</extra>'
          })
          summaries.push(`${name}: y = ${fit.slope.toFixed(6)} * x + ${fit.intercept.toFixed(6)} (R^2 = ${fit.rSquared.toFixed(6)})`)
          annotationLines.push(...linearStatsLines(config, plot, name, fit.slope, fit.intercept, fit.rSquared))
        } catch (error) {
          errors.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
    } else if (plot.render_mode === 'steps') {
      data.push(lineTrace(name, xValues, yValues, plot, index, 'lines+markers', stepShape(plot)))
    } else {
      data.push(lineTrace(name, xValues, yValues, plot, index, 'lines+markers'))
    }
  })

  formulaCurves.forEach(({ curve, index }) => {
    data.push({
      type: 'scatter',
      mode: 'lines',
      name: traceName(curve.label),
      x: curve.xValues,
      y: curve.yValues,
      line: {
        color: curve.color || palette[index % palette.length],
        width: curve.linewidth,
        dash: dashStyle(curve.linestyle)
      }
    })
    const equation = formatMathAnnotation(curve.equation_latex)
    if (equation) {
      annotationLines.push(escapeHtml(equation))
    }
  })

  const legendInfoItems = (config.legend_info ?? []).filter((item) => item.label.trim())
  if (legendInfoItems.length > 0) {
    data.push(legendTextOnlyTrace(t.legendInfoDivider, '__legend-info-divider__', 5000))
    legendInfoItems.forEach((item, index) => {
      data.push(legendTextOnlyTrace(item.label, item.id, 5001 + index))
    })
  }

  return {
    data,
    summaries,
    errors,
    layout: baseLayout(config, allXValues, allYValues, annotationLines),
    recomputeTicks
  }
}

function baseLayout(config: EditablePlotConfig, xValues: number[], yValues: number[], annotationLines: string[] = []): any {
  const xExponent = Number(config.x_exponent ?? 1)
  const yExponent = Number(config.y_exponent ?? 1)

  const xForceMin = (config.x_start_at_zero || config.x_allow_negative === false) ? 0 : undefined
  const hasYMaximum = config.y_max !== undefined && config.y_max !== null
  const yForceMin = (config.y_start_at_zero || config.y_allow_negative === false || hasYMaximum) ? 0 : undefined
  const yForceMax = hasYMaximum ? Number(config.y_max) : undefined

  const xFormatting: AxisFormatting = { step: config.x_tick_step, precision: config.x_tick_precision }
  const yFormatting: AxisFormatting = { step: config.y_tick_step, precision: config.y_tick_precision }

  const xTicks = buildTicks(xValues, xExponent, xForceMin, undefined, xFormatting)
  const yTicks = buildTicks(yValues, yExponent, yForceMin, yForceMax, yFormatting)

  // The first x-axis tick label sits at the y-axis line (left edge of the plot).
  // When both axes have labels this corner label overlaps the y-axis tick labels,
  // so suppress it. The tick mark remains; hover still shows the exact value.
  const xTicksAdjusted = (xTicks.ticktext?.length ?? 0) > 1 && (yTicks.tickvals?.length ?? 0) > 0
    ? { ...xTicks, ticktext: ['', ...(xTicks.ticktext ?? []).slice(1)] }
    : xTicks

  const annotations = annotationLines.length
    ? [
      {
        xref: 'paper',
        yref: 'paper',
        ...annotationPosition(config.stats_pos),
        text: annotationLines.join('<br>'),
        showarrow: false,
        align: 'left',
        bgcolor: 'rgba(255,255,255,0.92)',
        bordercolor: '#111827',
        borderwidth: 1,
        borderpad: 8,
        font: { color: '#111827', size: 13 }
      }
    ]
    : []

  return {
    title: { text: formatPlotTitle(config.plot_name), x: 0.03, xanchor: 'left', y: 1.34, yanchor: 'top', font: { size: 16 } },
    margin: { l: 96, r: 44, t: config.plot_name ? 152 : 92, b: 92 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    hovermode: 'closest',
    legend: { orientation: 'h', x: 0, y: 1.16, xanchor: 'left', yanchor: 'top' },
    xaxis: {
      title: { text: axisLabelWithUnit(config.x_label || 'x', config.x_unit || '', xExponent), standoff: 18 },
      automargin: true,
      gridcolor: '#e5e7eb',
      zerolinecolor: '#cbd5e1',
      ...xTicksAdjusted
    },
    yaxis: {
      title: { text: axisLabelWithUnit(config.y_label || 'y', config.y_unit || '', yExponent), standoff: 22 },
      automargin: true,
      gridcolor: '#e5e7eb',
      zerolinecolor: '#cbd5e1',
      ...yTicks
    },
    annotations
  }
}