export type PlotMode = 'fit' | 'lines' | 'steps'
export type PlotSourceType = 'workbook' | 'formula'
export type DataPlotMode = 'lines' | 'steps' | 'fit'
export type FitModel = 'linear' | 'polynomial' | 'exponential' | 'gaussian'
export type StepWhere = 'pre' | 'post' | 'mid'
export type StatsPosition = 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left'

export interface SeriesConfig {
  label?: string
  x_col_index?: number
  y_col_index?: number
  row_start?: number | null
  row_end?: number | null
  color?: string | null
  linestyle?: string
  marker?: string | null
  linewidth?: number
  step_where?: StepWhere
  scatter?: boolean
}

export interface FitConfig extends SeriesConfig {
  scatter?: boolean
}

export interface FormulaCurveConfig {
  label?: string
  expression?: string
  equation_latex?: string
  parameters?: Record<string, number>
  x_min?: number | null
  x_max?: number | null
  num_points?: number
  color?: string | null
  linestyle?: string
  linewidth?: number
}

export interface LegendInfoItem {
  id: string
  label: string
  color?: string | null
  linestyle?: string
  marker?: string | null
  linewidth?: number
}

export interface WorkbookPlotConfig extends SeriesConfig {
  id: string
  source_type: 'workbook'
  render_mode: DataPlotMode
  fit_model?: FitModel
  polynomial_degree?: number
  force_through_origin?: boolean
  start_at_zero?: boolean
  extrapolate?: boolean
  show_points?: boolean
  show_stats?: boolean
  show_slope?: boolean
  show_intercept?: boolean
  show_r_squared?: boolean
  show_gaussian_baseline?: boolean
  show_gaussian_amplitude?: boolean
  show_gaussian_mean?: boolean
  show_gaussian_std?: boolean
  show_polynomial_coefficients?: boolean
  show_exponential_coefficient?: boolean
  show_exponential_rate?: boolean
}

export interface FormulaPlotConfig extends FormulaCurveConfig {
  id: string
  source_type: 'formula'
}

export type PlotItemConfig = WorkbookPlotConfig | FormulaPlotConfig

export interface PlotTemplate {
  name: string
  plot_mode?: PlotMode
  plot_name?: string
  plot_label?: string
  x_col_index?: number
  y_col_index?: number
  x_label?: string
  y_label?: string
  x_unit?: string
  y_unit?: string
  x_exponent?: number
  y_exponent?: number
  slope_label?: string
  slope_unit?: string
  slope_exponent?: number
  slope_precision?: number
  intercept_label?: string
  intercept_unit?: string
  intercept_exponent?: number
  intercept_precision?: number
  show_slope?: boolean
  show_intercept?: boolean
  stats_pos?: StatsPosition
  force_through_origin?: boolean
  x_allow_negative?: boolean
  y_allow_negative?: boolean
  x_start_at_zero?: boolean
  y_start_at_zero?: boolean
  y_max?: number | null
  hide_base_series_legend?: boolean
  series_scatter_only?: boolean
  step_where?: StepWhere
  series?: SeriesConfig[]
  fits?: FitConfig[]
  formula_curves?: FormulaCurveConfig[]
  legend_info?: LegendInfoItem[]
}

export interface TemplatesFile {
  templates: PlotTemplate[]
}

export interface EditablePlotConfig extends PlotTemplate {
  plots: PlotItemConfig[]
  series: SeriesConfig[]
  fits: FitConfig[]
  formula_curves: FormulaCurveConfig[]
  legend_info: LegendInfoItem[]
}

export interface WorkbookColumn {
  index: number
  name: string
}

export interface WorksheetData {
  name: string
  columns: WorkbookColumn[]
  rows: unknown[][]
}

export interface WorkbookData {
  fileName: string
  sheets: WorksheetData[]
}
