import { replace as replaceLatexWithUnicode } from 'unicodeit'
import type { FormulaCurveConfig, SeriesConfig, WorksheetData } from '../types/templates'

export interface NumericSeries {
  xValues: number[]
  yValues: number[]
}

export interface LinearFitResult {
  slope: number
  intercept: number
  rSquared: number
}

export interface GaussianFitResult {
  baseline: number
  amplitude: number
  mean: number
  sigma: number
  rSquared: number
}

export interface PolynomialFitResult {
  coefficients: number[]
  rSquared: number
}

export interface ExponentialFitResult {
  coefficient: number
  rate: number
  rSquared: number
}

export interface FormulaCurveResult {
  label: string
  xValues: number[]
  yValues: number[]
  color?: string | null
  linestyle?: string
  linewidth: number
  equation_latex?: string
}

export function looksLikeMath(value: string): boolean {
  const trimmed = (value || '').trim()
  return trimmed.startsWith('\\') || /[\\_^{}]/.test(trimmed) || (trimmed.startsWith('$') && trimmed.endsWith('$'))
}

const superscriptCharacters: Record<string, string> = {
  '0': '⁰',
  '1': '¹',
  '2': '²',
  '3': '³',
  '4': '⁴',
  '5': '⁵',
  '6': '⁶',
  '7': '⁷',
  '8': '⁸',
  '9': '⁹',
  '+': '⁺',
  '-': '⁻',
  '=': '⁼',
  '(': '⁽',
  ')': '⁾',
  n: 'ⁿ',
  i: 'ⁱ',
  x: 'ˣ'
}

const subscriptCharacters: Record<string, string> = {
  '0': '₀',
  '1': '₁',
  '2': '₂',
  '3': '₃',
  '4': '₄',
  '5': '₅',
  '6': '₆',
  '7': '₇',
  '8': '₈',
  '9': '₉',
  '+': '₊',
  '-': '₋',
  '=': '₌',
  '(': '₍',
  ')': '₎',
  a: 'ₐ',
  e: 'ₑ',
  h: 'ₕ',
  i: 'ᵢ',
  j: 'ⱼ',
  k: 'ₖ',
  l: 'ₗ',
  m: 'ₘ',
  n: 'ₙ',
  o: 'ₒ',
  p: 'ₚ',
  r: 'ᵣ',
  s: 'ₛ',
  t: 'ₜ',
  u: 'ᵤ',
  v: 'ᵥ',
  x: 'ₓ'
}

function scriptText(value: string, map: Record<string, string>): string {
  return value.split('').map((character) => map[character] ?? character).join('')
}

function scriptGroupText(value: string, map: Record<string, string>): string {
  const cleanValue = value.replace(/[{}^_]/g, '')
  return scriptText(cleanValue, map)
}

function formatScriptSyntax(value: string): string {
  let text = value
  let previousText = ''
  while (text !== previousText) {
    previousText = text
    text = text
      .replace(/\^\{([^{}]+)\}/g, (_match, content: string) => scriptGroupText(content, superscriptCharacters))
      .replace(/_\{([^{}]+)\}/g, (_match, content: string) => scriptGroupText(content, subscriptCharacters))
  }

  return text
    .replace(/\^([0-9A-Za-z+\-=()])/g, (_match, content: string) => scriptText(content, superscriptCharacters))
    .replace(/_([0-9A-Za-z+\-=()])/g, (_match, content: string) => scriptText(content, subscriptCharacters))
}

export function formatLatexText(value?: string): string {
  let text = (value || '').trim()
  if (!text) {
    return ''
  }
  if (text.startsWith('$') && text.endsWith('$')) {
    text = text.slice(1, -1)
  }

  text = text
    .replace(/\\mathrm\{([^{}]*)\}/g, '$1')
    .replace(/\\text\{([^{}]*)\}/g, '$1')
    .replace(/\\frac\{([^{}]*)\}\{([^{}]*)\}/g, '($1)/($2)')
    .replace(/\\left|\\right/g, '')
    .replace(/\\[,;:!]/g, ' ')
    .replace(/~/g, ' ')
    .replace(/\s+/g, ' ')

  text = formatScriptSyntax(text)
  text = replaceLatexWithUnicode(text)
  text = text.replace(/\\([A-Za-z]+)/g, (_match, command: string) => command)
  text = text.replace(/[{}]/g, '')
  return text.trim()
}

export function labelInMathContext(value: string): string {
  const trimmed = (value || '').trim()
  if (trimmed.startsWith('$') && trimmed.endsWith('$')) {
    return trimmed.slice(1, -1)
  }
  if (looksLikeMath(trimmed)) {
    return trimmed
  }
  return `\\mathrm{${trimmed.replace(/ /g, '\\ ')}}`
}

export function axisLabelWithUnit(label: string, unit: string, exponent = 1): string {
  const cleanLabel = formatLatexText(label)
  const cleanUnit = formatLatexText(unit)
  const exponentText = scriptText(String(exponent), superscriptCharacters)

  if (exponent !== 0 && cleanUnit) {
    return `${cleanLabel} (10${exponentText} · ${cleanUnit})`
  }
  if (exponent !== 0) {
    return `${cleanLabel} · 10${exponentText}`
  }
  if (cleanUnit) {
    return `${cleanLabel} (${cleanUnit})`
  }
  return cleanLabel
}

export function formatPlotTitle(title?: string): string {
  return formatLatexText(title)
}

export function formatMathAnnotation(expression?: string): string {
  return formatLatexText(expression)
}

export function extractNumericSeries(sheet: WorksheetData, config: SeriesConfig): NumericSeries {
  const xColumn = config.x_col_index
  const yColumn = config.y_col_index
  if (xColumn === undefined || yColumn === undefined) {
    throw new Error('Choose both X and Y columns.')
  }
  if (xColumn < 0 || xColumn >= sheet.columns.length || yColumn < 0 || yColumn >= sheet.columns.length) {
    throw new Error('Selected columns are outside the uploaded sheet.')
  }

  const start = config.row_start ?? 0
  const end = config.row_end ?? sheet.rows.length
  const selectedRows = sheet.rows.slice(Math.max(0, start), end === null ? sheet.rows.length : Math.max(0, end))
  const xValues: number[] = []
  const yValues: number[] = []

  for (const row of selectedRows) {
    const rawXValue = row[xColumn]
    const rawYValue = row[yColumn]
    if (rawXValue === null || rawXValue === undefined || rawXValue === '' || rawYValue === null || rawYValue === undefined || rawYValue === '') {
      continue
    }
    const xValue = Number(rawXValue)
    const yValue = Number(rawYValue)
    if (Number.isFinite(xValue) && Number.isFinite(yValue)) {
      xValues.push(xValue)
      yValues.push(yValue)
    }
  }

  if (xValues.length < 2) {
    throw new Error('Need at least two valid numeric X/Y pairs.')
  }

  return { xValues, yValues }
}

export function linearFit(xValues: number[], yValues: number[], forceThroughOrigin = false): LinearFitResult {
  if (xValues.length !== yValues.length || xValues.length < 2) {
    throw new Error('Linear fit requires matching X/Y arrays with at least two points.')
  }

  let slope = 0
  let intercept = 0
  if (forceThroughOrigin) {
    const denominator = xValues.reduce((sum, value) => sum + value * value, 0)
    if (denominator === 0) {
      throw new Error('Cannot force an origin fit when all X values are zero.')
    }
    slope = xValues.reduce((sum, value, index) => sum + value * yValues[index], 0) / denominator
  } else {
    const count = xValues.length
    const sumX = xValues.reduce((sum, value) => sum + value, 0)
    const sumY = yValues.reduce((sum, value) => sum + value, 0)
    const sumXX = xValues.reduce((sum, value) => sum + value * value, 0)
    const sumXY = xValues.reduce((sum, value, index) => sum + value * yValues[index], 0)
    const denominator = count * sumXX - sumX * sumX
    if (denominator === 0) {
      throw new Error('Cannot fit a line when all X values are identical.')
    }
    slope = (count * sumXY - sumX * sumY) / denominator
    intercept = (sumY - slope * sumX) / count
  }

  const meanY = yValues.reduce((sum, value) => sum + value, 0) / yValues.length
  const residualSum = yValues.reduce((sum, value, index) => {
    const fitted = slope * xValues[index] + intercept
    return sum + (value - fitted) ** 2
  }, 0)
  const totalSum = yValues.reduce((sum, value) => sum + (value - meanY) ** 2, 0)
  const rSquared = totalSum > 0 ? 1 - residualSum / totalSum : Number.NaN

  return { slope, intercept, rSquared }
}

function rSquaredForValues(yValues: number[], fittedValues: number[]): number {
  const meanY = yValues.reduce((sum, value) => sum + value, 0) / yValues.length
  const residualSum = yValues.reduce((sum, value, index) => sum + (value - fittedValues[index]) ** 2, 0)
  const totalSum = yValues.reduce((sum, value) => sum + (value - meanY) ** 2, 0)
  return totalSum > 0 ? 1 - residualSum / totalSum : Number.NaN
}

function solveLinearSystem(matrix: number[][], vector: number[]): number[] {
  const size = vector.length
  const augmented = matrix.map((row, index) => [...row, vector[index]])

  for (let column = 0; column < size; column += 1) {
    let pivotRow = column
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivotRow][column])) {
        pivotRow = row
      }
    }
    if (Math.abs(augmented[pivotRow][column]) < 1e-12) {
      throw new Error('Could not solve the fit because the data matrix is singular.')
    }
    [augmented[column], augmented[pivotRow]] = [augmented[pivotRow], augmented[column]]

    const pivot = augmented[column][column]
    for (let entry = column; entry <= size; entry += 1) {
      augmented[column][entry] /= pivot
    }
    for (let row = 0; row < size; row += 1) {
      if (row === column) {
        continue
      }
      const factor = augmented[row][column]
      for (let entry = column; entry <= size; entry += 1) {
        augmented[row][entry] -= factor * augmented[column][entry]
      }
    }
  }

  return augmented.map((row) => row[size])
}

export function polynomialValue(xValue: number, coefficients: number[]): number {
  return coefficients.reduce((sum, coefficient, power) => sum + coefficient * xValue ** power, 0)
}

export function polynomialFit(xValues: number[], yValues: number[], degree = 2): PolynomialFitResult {
  if (xValues.length !== yValues.length || xValues.length < 2) {
    throw new Error('Polynomial fit requires matching X/Y arrays with at least two points.')
  }
  const safeDegree = Math.max(1, Math.min(8, Math.trunc(Number(degree) || 1)))
  if (xValues.length < safeDegree + 1) {
    throw new Error(`Polynomial degree ${safeDegree} requires at least ${safeDegree + 1} points.`)
  }

  const size = safeDegree + 1
  const matrix = Array.from({ length: size }, (_row, row) => (
    Array.from({ length: size }, (_column, column) => xValues.reduce((sum, value) => sum + value ** (row + column), 0))
  ))
  const vector = Array.from({ length: size }, (_unused, row) => xValues.reduce((sum, value, index) => sum + yValues[index] * value ** row, 0))
  const coefficients = solveLinearSystem(matrix, vector)
  const fittedValues = xValues.map((value) => polynomialValue(value, coefficients))

  return { coefficients, rSquared: rSquaredForValues(yValues, fittedValues) }
}

export function exponentialValue(xValue: number, fit: Pick<ExponentialFitResult, 'coefficient' | 'rate'>): number {
  return fit.coefficient * Math.exp(fit.rate * xValue)
}

export function exponentialFit(xValues: number[], yValues: number[]): ExponentialFitResult {
  if (xValues.length !== yValues.length || xValues.length < 2) {
    throw new Error('Exponential fit requires matching X/Y arrays with at least two points.')
  }
  if (yValues.some((value) => value <= 0 || !Number.isFinite(value))) {
    throw new Error('Exponential fit requires positive finite Y values.')
  }
  const logYValues = yValues.map((value) => Math.log(value))
  const logFit = linearFit(xValues, logYValues, false)
  const coefficient = Math.exp(logFit.intercept)
  const rate = logFit.slope
  const fittedValues = xValues.map((value) => exponentialValue(value, { coefficient, rate }))

  return { coefficient, rate, rSquared: rSquaredForValues(yValues, fittedValues) }
}

export function gaussianValue(xValue: number, fit: Pick<GaussianFitResult, 'baseline' | 'amplitude' | 'mean' | 'sigma'>): number {
  return fit.baseline + fit.amplitude * Math.exp(-((xValue - fit.mean) ** 2) / (2 * fit.sigma ** 2))
}

export function gaussianFit(xValues: number[], yValues: number[]): GaussianFitResult {
  if (xValues.length !== yValues.length || xValues.length < 3) {
    throw new Error('Gaussian fit requires matching X/Y arrays with at least three points.')
  }

  const finitePairs = xValues
    .map((xValue, index) => ({ xValue, yValue: yValues[index] }))
    .filter((pair) => Number.isFinite(pair.xValue) && Number.isFinite(pair.yValue))
  if (finitePairs.length < 3) {
    throw new Error('Gaussian fit requires at least three finite points.')
  }

  const xs = finitePairs.map((pair) => pair.xValue)
  const ys = finitePairs.map((pair) => pair.yValue)
  const minimumX = Math.min(...xs)
  const maximumX = Math.max(...xs)
  const minimumY = Math.min(...ys)
  const maximumY = Math.max(...ys)
  const peakIndex = ys.indexOf(maximumY)
  const rangeX = Math.max(maximumX - minimumX, Number.EPSILON)
  const initialAmplitude = maximumY - minimumY
  if (initialAmplitude === 0) {
    throw new Error('Gaussian fit requires varying Y values.')
  }

  let baseline = minimumY
  let amplitude = initialAmplitude
  let mean = xs[peakIndex]
  let sigma = rangeX / 4

  const weights = ys.map((value) => Math.max(value - baseline, 0))
  const weightSum = weights.reduce((sum, value) => sum + value, 0)
  if (weightSum > 0) {
    mean = xs.reduce((sum, value, index) => sum + value * weights[index], 0) / weightSum
    const variance = xs.reduce((sum, value, index) => sum + (value - mean) ** 2 * weights[index], 0) / weightSum
    sigma = Math.sqrt(Math.max(variance, Number.EPSILON))
  }

  function sumSquaredError(params: [number, number, number, number]): number {
    const [nextBaseline, nextAmplitude, nextMean, nextSigma] = params
    if (nextSigma <= 0 || !params.every(Number.isFinite)) {
      return Number.POSITIVE_INFINITY
    }
    return xs.reduce((sum, value, index) => {
      const fitted = gaussianValue(value, {
        baseline: nextBaseline,
        amplitude: nextAmplitude,
        mean: nextMean,
        sigma: nextSigma
      })
      return sum + (ys[index] - fitted) ** 2
    }, 0)
  }

  let params: [number, number, number, number] = [baseline, amplitude, mean, sigma]
  let steps: [number, number, number, number] = [
    Math.max(Math.abs(amplitude) * 0.2, 0.1),
    Math.max(Math.abs(amplitude) * 0.2, 0.1),
    Math.max(rangeX * 0.1, 0.1),
    Math.max(Math.abs(sigma) * 0.2, 0.1)
  ]
  let bestError = sumSquaredError(params)

  for (let iteration = 0; iteration < 160; iteration += 1) {
    let improved = false
    for (let paramIndex = 0; paramIndex < params.length; paramIndex += 1) {
      for (const direction of [-1, 1]) {
        const candidate = [...params] as [number, number, number, number]
        candidate[paramIndex] += steps[paramIndex] * direction
        if (paramIndex === 3) {
          candidate[paramIndex] = Math.max(candidate[paramIndex], Number.EPSILON)
        }
        const candidateError = sumSquaredError(candidate)
        if (candidateError < bestError) {
          params = candidate
          bestError = candidateError
          improved = true
        }
      }
    }
    if (!improved) {
      steps = steps.map((step) => step * 0.5) as [number, number, number, number]
      if (Math.max(...steps) < 1e-8) {
        break
      }
    }
  }

  [baseline, amplitude, mean, sigma] = params
  const meanY = ys.reduce((sum, value) => sum + value, 0) / ys.length
  const totalSum = ys.reduce((sum, value) => sum + (value - meanY) ** 2, 0)
  const rSquared = totalSum > 0 ? 1 - bestError / totalSum : Number.NaN

  return { baseline, amplitude, mean, sigma, rSquared }
}

export function linspace(minimum: number, maximum: number, count: number): number[] {
  if (count < 2) {
    return [minimum]
  }
  const step = (maximum - minimum) / (count - 1)
  return Array.from({ length: count }, (_unused, index) => minimum + step * index)
}

export function convertFormulaExpression(expression: string): string {
  return normalizeFormulaExpression(expression)
    .replace(/np\./g, '')
    .replace(/\bpi\b/g, 'pi')
    .replace(/\be\b/g, 'e')
    .replace(/\*\*/g, '^')
}

function replaceLatexGroups(expression: string, pattern: RegExp, replacer: (first: string, second?: string) => string): string {
  let nextExpression = expression
  let previousExpression = ''
  while (nextExpression !== previousExpression) {
    previousExpression = nextExpression
    nextExpression = nextExpression.replace(pattern, (_match, first: string, second?: string) => replacer(first, second))
  }
  return nextExpression
}

function normalizeFormulaExpression(expression: string): string {
  let normalized = (expression || '').trim()
  if (normalized.startsWith('$') && normalized.endsWith('$')) {
    normalized = normalized.slice(1, -1).trim()
  }

  normalized = normalized
    .replace(/^\s*(?:y|f\s*\(\s*x\s*\))\s*=/i, '')
    .replace(/\\left|\\right/g, '')
    .replace(/\\,/g, ' ')
    .replace(/\\cdot|\\times/g, '*')
    .replace(/\\frac\b/g, 'frac')
    .replace(/\\pi\b/g, 'pi')
    .replace(/\\exp\b/g, 'exp')
    .replace(/\\ln\b/g, 'log')
    .replace(/\\log\b/g, 'log')
    .replace(/\\sin\b/g, 'sin')
    .replace(/\\cos\b/g, 'cos')
    .replace(/\\tan\b/g, 'tan')
    .replace(/\\sqrt\b/g, 'sqrt')

  normalized = replaceLatexGroups(normalized, /sqrt\s*\{([^{}]+)\}/g, (value) => `sqrt(${value})`)
  normalized = replaceLatexGroups(normalized, /frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}/g, (numerator, denominator) => `((${numerator})/(${denominator}))`)
  normalized = replaceLatexGroups(normalized, /\be\s*\^\s*\{([^{}]+)\}/g, (exponent) => `exp(${exponent})`)
  normalized = replaceLatexGroups(normalized, /\^\s*\{([^{}]+)\}/g, (exponent) => `^(${exponent})`)
  normalized = normalized.replace(/\}\s*\{/g, ')*(').replace(/[{}]/g, '')

  normalized = normalized
    .replace(/(\d(?:\.\d+)?|\)|\bpi\b|\be\b)\s+(?=\d|[A-Za-z_(])/g, '$1*')
    .replace(/(\d(?:\.\d+)?|\)|\bpi\b|\be\b)(?=[A-Za-z_(])/g, '$1*')
    .replace(/\)(?=\d|[A-Za-z_(])/g, ')*')
    .replace(/\s+/g, ' ')
    .trim()

  normalized = normalized.replace(/(^|[+\-*/%(,])\s*-\s*([A-Za-z_]\w*|\d+(?:\.\d+)?|\([^()]+\))\s*\^\s*([A-Za-z_]\w*|\d+(?:\.\d+)?|\([^()]+\))/g, '$1-(($2)^($3))')
  return normalized
}

const allowedFunctions = new Set(['abs', 'acos', 'asin', 'atan', 'cos', 'exp', 'log', 'max', 'min', 'pow', 'sin', 'sqrt', 'tan'])
const allowedConstants = new Set(['pi', 'e'])

function assertSafeParameterName(name: string): void {
  if (!/^[A-Za-z_]\w*$/.test(name) || allowedFunctions.has(name) || allowedConstants.has(name) || name === 'x') {
    throw new Error(`Invalid formula parameter name: ${name}`)
  }
}

function assertNoUnsafeDots(expression: string): void {
  for (let index = 0; index < expression.length; index += 1) {
    if (expression[index] !== '.') {
      continue
    }
    const previous = expression[index - 1] ?? ''
    const next = expression[index + 1] ?? ''
    if (!/\d/.test(previous) || !/\d/.test(next)) {
      throw new Error('Formula expressions may not access object properties.')
    }
  }
}

function compileSafeFormula(expression: string, parameterNames: string[]): (...values: number[]) => number {
  const normalized = normalizeFormulaExpression(expression).replace(/np\./g, '').replace(/\^/g, '**')
  if (!/^[0-9A-Za-z_+\-*/%().,\s]*$/.test(normalized)) {
    throw new Error('Formula expression contains unsupported characters.')
  }
  assertNoUnsafeDots(normalized)

  const knownParameters = new Set(parameterNames)
  let jsExpression = normalized.replace(/\b[A-Za-z_]\w*\b/g, (identifier) => {
    if (identifier === 'x') {
      return identifier
    }
    if (allowedConstants.has(identifier)) {
      return identifier === 'pi' ? 'Math.PI' : 'Math.E'
    }
    if (allowedFunctions.has(identifier)) {
      return `Math.${identifier}`
    }
    if (knownParameters.has(identifier)) {
      return identifier
    }
    throw new Error(`Unsupported formula identifier: ${identifier}`)
  })

  jsExpression = jsExpression.replace(/\*\*\*\*/g, '**')
  return Function('x', ...parameterNames, `'use strict'; return (${jsExpression});`) as (...values: number[]) => number
}

export function buildFormulaCurve(
  curve: FormulaCurveConfig,
  fallbackMinimum: number | null,
  fallbackMaximum: number | null
): FormulaCurveResult {
  const expression = (curve.expression || '').trim()
  if (!expression) {
    throw new Error('Formula curves need an expression.')
  }

  const sampleCount = Math.max(2, Number(curve.num_points ?? 400))
  const minimum = Number(curve.x_min ?? fallbackMinimum)
  const maximum = Number(curve.x_max ?? fallbackMaximum)
  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || maximum <= minimum) {
    throw new Error('Formula curves need a valid X range.')
  }

  const xValues = linspace(minimum, maximum, sampleCount)
  const parameters = curve.parameters ?? {}
  const parameterNames = Object.keys(parameters)
  parameterNames.forEach(assertSafeParameterName)
  const evaluator = compileSafeFormula(expression, parameterNames)
  const yValues = xValues.map((xValue) => {
    const yValue = Number(evaluator(xValue, ...parameterNames.map((name) => parameters[name])))
    if (!Number.isFinite(yValue)) {
      throw new Error('Formula expression produced a non-finite value.')
    }
    return yValue
  })

  return {
    label: curve.label || 'Theoretical curve',
    xValues,
    yValues,
    color: curve.color,
    linestyle: curve.linestyle,
    linewidth: Number(curve.linewidth ?? 2),
    equation_latex: curve.equation_latex
  }
}

export function scaledTickText(value: number, exponent: number, precision?: number | null): string {
  const divisor = 10 ** exponent
  const display = divisor === 0 || divisor === 1 ? value : value / divisor
  if (precision !== undefined && precision !== null && Number.isFinite(precision)) {
    const decimals = Math.max(0, Math.min(20, Math.trunc(precision)))
    return display.toFixed(decimals)
  }
  return Number.isInteger(display) ? String(display) : Number(display.toPrecision(6)).toString()
}
