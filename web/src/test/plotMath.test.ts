import { describe, expect, it } from 'vitest'
import {
  axisLabelWithUnit,
  buildFormulaCurve,
  convertFormulaExpression,
  exponentialFit,
  exponentialValue,
  extractNumericSeries,
  formatLatexText,
  gaussianFit,
  gaussianValue,
  linearFit,
  polynomialFit,
  polynomialValue
} from '../lib/plotMath'
import type { WorksheetData } from '../types/templates'

const sheet: WorksheetData = {
  name: 'Sheet1',
  columns: [
    { index: 0, name: 'x' },
    { index: 1, name: 'y' }
  ],
  rows: [
    [1, 3],
    [2, 5],
    [3, 7],
    ['bad', 9]
  ]
}

describe('plot math', () => {
  it('extracts finite numeric pairs', () => {
    const series = extractNumericSeries(sheet, { x_col_index: 0, y_col_index: 1 })
    expect(series.xValues).toEqual([1, 2, 3])
    expect(series.yValues).toEqual([3, 5, 7])
  })

  it('skips blank cells instead of treating them as zeroes', () => {
    const series = extractNumericSeries(
      {
        ...sheet,
        rows: [
          [1, 2],
          [2, null],
          ['', 4],
          [3, 8]
        ]
      },
      { x_col_index: 0, y_col_index: 1 }
    )

    expect(series.xValues).toEqual([1, 3])
    expect(series.yValues).toEqual([2, 8])
  })

  it('computes linear and forced-origin fits', () => {
    expect(linearFit([1, 2, 3], [3, 5, 7])).toMatchObject({ slope: 2, intercept: 1 })
    expect(linearFit([1, 2], [2, 4], true)).toMatchObject({ slope: 2, intercept: 0 })
  })

  it('fits gaussian curves from sampled data', () => {
    const xValues = [-3, -2, -1, 0, 1, 2, 3]
    const yValues = xValues.map((value) => gaussianValue(value, { baseline: 0.5, amplitude: 4, mean: 0.4, sigma: 1.2 }))
    const fit = gaussianFit(xValues, yValues)
    expect(fit.baseline).toBeCloseTo(0.5, 1)
    expect(fit.amplitude).toBeCloseTo(4, 1)
    expect(fit.mean).toBeCloseTo(0.4, 1)
    expect(fit.sigma).toBeCloseTo(1.2, 1)
    expect(fit.rSquared).toBeGreaterThan(0.99)
  })

  it('fits polynomial curves from sampled data', () => {
    const xValues = [-2, -1, 0, 1, 2, 3]
    const yValues = xValues.map((value) => 1 - 2 * value + 3 * value ** 2)
    const fit = polynomialFit(xValues, yValues, 2)

    expect(fit.coefficients[0]).toBeCloseTo(1, 8)
    expect(fit.coefficients[1]).toBeCloseTo(-2, 8)
    expect(fit.coefficients[2]).toBeCloseTo(3, 8)
    expect(polynomialValue(4, fit.coefficients)).toBeCloseTo(41, 8)
    expect(fit.rSquared).toBeCloseTo(1, 8)
  })

  it('fits exponential curves from sampled data', () => {
    const xValues = [0, 1, 2, 3, 4]
    const yValues = xValues.map((value) => 2.5 * Math.exp(0.4 * value))
    const fit = exponentialFit(xValues, yValues)

    expect(fit.coefficient).toBeCloseTo(2.5, 8)
    expect(fit.rate).toBeCloseTo(0.4, 8)
    expect(exponentialValue(2, fit)).toBeCloseTo(yValues[2], 8)
    expect(fit.rSquared).toBeCloseTo(1, 8)
  })

  it('formats LaTeX-like labels as visible plot text', () => {
    expect(formatLatexText('$\\mu_0^2 \\cdot x$')).toBe('μ₀² ⋅ x')
    expect(formatLatexText('\\phi')).toBe('ϕ')
    expect(formatLatexText('\\varphi')).toBe('φ')
    expect(formatLatexText('\\Phi')).toBe('Φ')
    expect(formatLatexText('y=e^{-x^2}')).toBe('y=e⁻ˣ²')
    expect(axisLabelWithUnit('a^2', 'm^2', -3)).toBe('a² (10⁻³ · m²)')
  })

  it('converts and evaluates NumPy-like formula curves safely', () => {
    expect(convertFormulaExpression('np.sqrt(x) + x**2')).toBe('sqrt(x) + x^2')
    const curve = buildFormulaCurve(
      { expression: 'np.sqrt(x) + a', parameters: { a: 2 }, x_min: 1, x_max: 4, num_points: 2 },
      null,
      null
    )
    expect(curve.yValues).toEqual([3, 4])
  })

  it('evaluates formula powers after unary minus', () => {
    const curve = buildFormulaCurve(
      { expression: 'exp(-x^2)', x_min: 0, x_max: 2, num_points: 3 },
      null,
      null
    )

    expect(curve.yValues[0]).toBeCloseTo(1, 8)
    expect(curve.yValues[1]).toBeCloseTo(Math.exp(-1), 8)
    expect(curve.yValues[2]).toBeCloseTo(Math.exp(-4), 8)
  })

  it('evaluates LaTeX-like formula expressions', () => {
    const curve = buildFormulaCurve(
      { expression: 'y=e^{-x^2}+\\frac{1}{2}x', x_min: 0, x_max: 2, num_points: 3 },
      null,
      null
    )

    expect(curve.yValues[0]).toBeCloseTo(1, 8)
    expect(curve.yValues[1]).toBeCloseTo(Math.exp(-1) + 0.5, 8)
    expect(curve.yValues[2]).toBeCloseTo(Math.exp(-4) + 1, 8)
  })
})
