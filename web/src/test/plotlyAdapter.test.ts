import { describe, expect, it } from 'vitest'
import { buildPlotFigure } from '../lib/plotlyAdapter'
import { gaussianValue } from '../lib/plotMath'
import type { EditablePlotConfig, WorksheetData } from '../types/templates'

const xValues = [-3, -2, -1, 0, 1, 2, 3]
const sheet: WorksheetData = {
  name: 'Sheet1',
  columns: [
    { index: 0, name: 'x' },
    { index: 1, name: 'y' }
  ],
  rows: xValues.map((value) => [value, gaussianValue(value, { baseline: 0.5, amplitude: 4, mean: 0.25, sigma: 1.15 })])
}

function configWithPlot(overrides: Partial<EditablePlotConfig['plots'][number]> = {}): EditablePlotConfig {
  return {
    name: 'Test',
    plot_name: 'Demo',
    x_label: 'Distance',
    y_label: 'Signal',
    x_unit: 'm',
    y_unit: 'V',
    x_exponent: 0,
    y_exponent: 0,
    slope_exponent: 0,
    intercept_exponent: 0,
    slope_precision: 5,
    intercept_precision: 5,
    stats_pos: 'top-left',
    plots: [
      {
        id: 'plot-1',
        source_type: 'workbook',
        render_mode: 'fit',
        fit_model: 'gaussian',
        label: 'Gaussian data',
        x_col_index: 0,
        y_col_index: 1,
        show_points: true,
        show_stats: true,
        show_gaussian_baseline: true,
        show_gaussian_amplitude: true,
        show_gaussian_mean: true,
        show_gaussian_std: true,
        show_r_squared: true,
        ...overrides
      }
    ],
    series: [],
    fits: [],
    formula_curves: [],
    legend_info: []
  }
}

describe('plotly adapter', () => {
  it('uses visible axis title objects and fit stats annotations', () => {
    const figure = buildPlotFigure(configWithPlot(), sheet)

    expect(figure.layout.xaxis.title.text).toBe('Distance (m)')
    expect(figure.layout.xaxis.automargin).toBe(true)
    expect(figure.layout.yaxis.title.text).toBe('Signal (V)')
    expect(figure.layout.yaxis.automargin).toBe(true)
    expect(figure.layout.annotations[0].text).toContain('Gaussian data')
    expect(figure.layout.annotations[0].text).toContain('σ')
    expect(figure.layout.annotations[0].text).toContain('R²')
    expect(figure.layout.margin.t).toBeGreaterThanOrEqual(120)
    expect(figure.layout.title.y).toBeGreaterThan(figure.layout.legend.y)
  })

  it('formats plot labels for legends and avoids duplicate fit legend entries', () => {
    const figure = buildPlotFigure(configWithPlot({ label: 'm_1 = \\Phi' }), sheet)

    expect(figure.data[0].name).toBe('m₁ = Φ')
    expect(figure.data[1].name).toBe('m₁ = Φ')
    expect(figure.data[0].showlegend).toBe(true)
    expect(figure.data[1].showlegend).toBe(false)
    expect(figure.data[0].name).not.toContain('data')
  })

  it('honors per-plot stats field toggles', () => {
    const figure = buildPlotFigure(configWithPlot({ show_gaussian_std: false, show_r_squared: false }), sheet)

    expect(figure.layout.annotations[0].text).not.toContain('σ')
    expect(figure.layout.annotations[0].text).not.toContain('R²')
    expect(figure.layout.annotations[0].text).toContain('A =')
  })

  it('builds polynomial fit traces and coefficient stats', () => {
    const polynomialSheet: WorksheetData = {
      ...sheet,
      rows: [-2, -1, 0, 1, 2].map((value) => [value, 1 + 2 * value + 3 * value ** 2])
    }
    const figure = buildPlotFigure(configWithPlot({ fit_model: 'polynomial', polynomial_degree: 2 }), polynomialSheet)

    expect(figure.errors).toEqual([])
    expect(figure.summaries[0]).toContain('x^2')
    expect(figure.layout.annotations[0].text).toContain('a₂')
  })

  it('builds exponential fit traces and parameter stats', () => {
    const exponentialSheet: WorksheetData = {
      ...sheet,
      rows: [0, 1, 2, 3].map((value) => [value, 2 * Math.exp(0.5 * value)])
    }
    const figure = buildPlotFigure(configWithPlot({ fit_model: 'exponential' }), exponentialSheet)

    expect(figure.errors).toEqual([])
    expect(figure.summaries[0]).toContain('exp')
    expect(figure.layout.annotations[0].text).toContain('A =')
    expect(figure.layout.annotations[0].text).toContain('b =')
  })

  it('adds legend-only info entries without extending plot data ranges', () => {
    const figure = buildPlotFigure({
      ...configWithPlot(),
      legend_info: [
        {
          id: 'info-1',
          label: 'calibrare: \\Phi_0'
        }
      ]
    }, sheet)

    const dividerTrace = figure.data[figure.data.length - 2]
    const infoTrace = figure.data[figure.data.length - 1]
    expect(dividerTrace.name).toContain('Legend Info')
    expect(dividerTrace.x).toEqual([null])
    expect(infoTrace.name).toBe('calibrare: Φ₀')
    expect(infoTrace.x).toEqual([null])
    expect(infoTrace.y).toEqual([null])
    expect(infoTrace.showlegend).toBe(true)
    expect(infoTrace.hoverinfo).toBe('skip')
    expect(infoTrace.line.width).toBe(0)
    expect(infoTrace.marker.size).toBe(0)
    expect(figure.layout.xaxis.tickvals).not.toContain(null)
  })
})
