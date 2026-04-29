import { ChevronDown, ChevronLeft, ChevronRight, Download, GripHorizontal, GripVertical, Plus, RefreshCw, Trash2, Upload } from 'lucide-react'
import { ChangeEvent, useMemo, useRef, useState } from 'react'
import PlotPreview, { type PlotPreviewHandle } from './components/PlotPreview'
import { detectInitialLanguage, getTranslations, languageOptions, localizeRuntimeMessage, type AppLanguage, type AppTranslations } from './lib/i18n'
import { formatLatexText } from './lib/plotMath'
import { buildPlotFigure } from './lib/plotlyAdapter'
import { buildWorkbookSettingsTemplate, parseWorkbookSettingsTemplate, templateToCsv } from './lib/templateTransfer'
import { addSheetColumn, addSheetRow, formatEditableCell, getSheet, parseEditableCell, parseWorkbook, updateCell, updateColumnName } from './lib/workbook'
import type {
  DataPlotMode,
  EditablePlotConfig,
  FitModel,
  FormulaPlotConfig,
  LegendInfoItem,
  PlotItemConfig,
  WorkbookData,
  WorkbookPlotConfig,
  WorksheetData
} from './types/templates'

const dataPlotModes: DataPlotMode[] = ['lines', 'steps', 'fit']
const fitModels: FitModel[] = ['linear', 'polynomial', 'exponential', 'gaussian']
const lineStyleOptions = ['-', '--', ':', '-.'] as const
const markerOptions = ['o', 's', '^', 'x', 'none'] as const
const stepOptions = ['pre', 'post', 'mid'] as const
const statsPositions = ['top-right', 'top-left', 'bottom-right', 'bottom-left'] as const
const minimumLeftPaneWeight = 18
const minimumRightPaneWeight = 18
const minimumMiddlePaneWeight = 28
const collapsedPaneSize = '44px'

function createId(): string {
  return `plot-${Date.now()}-${Math.random().toString(36).slice(2)}`
}

function createWorkbookPlot(index: number, t: AppTranslations, label = t.plot(index + 1)): WorkbookPlotConfig {
  return {
    id: createId(),
    source_type: 'workbook',
    render_mode: 'lines',
    fit_model: 'linear',
    polynomial_degree: 2,
    label,
    x_col_index: 0,
    y_col_index: 1,
    marker: 'o',
    linestyle: '-',
    linewidth: 2,
    show_points: true,
    force_through_origin: false,
    start_at_zero: false,
    extrapolate: false,
    show_stats: true,
    show_slope: true,
    show_intercept: true,
    show_r_squared: true,
    show_gaussian_baseline: true,
    show_gaussian_amplitude: true,
    show_gaussian_mean: true,
    show_gaussian_std: true,
    show_polynomial_coefficients: true,
    show_exponential_coefficient: true,
    show_exponential_rate: true
  }
}

function createFormulaPlot(index: number, t: AppTranslations, label = t.formulaItem(index + 1)): FormulaPlotConfig {
  return {
    id: createId(),
    source_type: 'formula',
    label,
    expression: 'x',
    equation_latex: '',
    x_min: null,
    x_max: null,
    num_points: 400,
    linestyle: '--',
    linewidth: 2
  }
}

function createLegendInfoItem(index: number, t: AppTranslations, label = t.infoItem(index + 1)): LegendInfoItem {
  return {
    id: createId(),
    label
  }
}

function createEmptyConfig(t: AppTranslations): EditablePlotConfig {
  return {
    name: t.interactivePlotName,
    plot_name: '',
    plot_mode: 'lines',
    x_label: 'x',
    y_label: 'y',
    x_exponent: 0,
    y_exponent: 0,
    slope_exponent: 0,
    intercept_exponent: 0,
    slope_precision: 5,
    intercept_precision: 5,
    show_slope: true,
    show_intercept: true,
    stats_pos: 'bottom-right',
    plots: [createWorkbookPlot(0, t)],
    series: [],
    fits: [],
    formula_curves: [],
    legend_info: []
  }
}

function sanitizeFileBaseName(value: string): string {
  return (value || 'plot-settings')
    .replace(/\.[^.]+$/, '')
    .replace(/[^a-z0-9_-]+/gi, '-')
    .replace(/^-+|-+$/g, '') || 'plot-settings'
}

function downloadTextFile(fileName: string, mimeType: string, content: string) {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }))
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.click()
  URL.revokeObjectURL(url)
}

function withFreshPlotIds(config: EditablePlotConfig, t: AppTranslations): EditablePlotConfig {
  const fallback = createEmptyConfig(t)
  return {
    ...fallback,
    ...config,
    plots: (config.plots?.length ? config.plots : fallback.plots).map((plot) => ({ ...plot, id: createId() }) as PlotItemConfig),
    series: config.series ?? [],
    fits: config.fits ?? [],
    formula_curves: config.formula_curves ?? [],
    legend_info: (config.legend_info ?? []).map((item) => ({ ...item, id: createId() }))
  }
}

function NumberInput({ label, value, onChange, placeholder }: { label: string; value?: number | null; onChange: (value: number | null) => void; placeholder?: string }) {
  return (
    <label className="field compact-field">
      <span>{label}</span>
      <input
        type="number"
        value={value ?? ''}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value === '' ? null : Number(event.target.value))}
      />
    </label>
  )
}

function TextInput({ label, value, onChange, placeholder }: { label: string; value?: string; onChange: (value: string) => void; placeholder?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input value={value ?? ''} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </label>
  )
}

function ColumnSelect({ label, value, sheet, onChange, t }: { label: string; value?: number; sheet: WorksheetData | null; onChange: (value: number) => void; t: AppTranslations }) {
  return (
    <label className="field compact-field">
      <span>{label}</span>
      <select value={value ?? ''} onChange={(event) => onChange(Number(event.target.value))} disabled={!sheet}>
        <option value="">{t.column}</option>
        {sheet?.columns.map((column) => (
          <option value={column.index} key={column.index}>{column.index}: {column.name}</option>
        ))}
      </select>
    </label>
  )
}

function PlotActions({ onAddWorkbook, onAddFormula, t }: { onAddWorkbook: () => void; onAddFormula: () => void; t: AppTranslations }) {
  return (
    <div className="button-row">
      <button className="secondary-button" type="button" onClick={onAddWorkbook}><Plus size={15} /><span>{t.data}</span></button>
      <button className="secondary-button" type="button" onClick={onAddFormula}><Plus size={15} /><span>{t.formula}</span></button>
    </div>
  )
}

function WorkbookPlotFields({ plot, sheet, onChange, t }: { plot: WorkbookPlotConfig; sheet: WorksheetData | null; onChange: (patch: Partial<WorkbookPlotConfig>) => void; t: AppTranslations }) {
  return (
    <>
      <TextInput label={t.label} value={plot.label} onChange={(value) => onChange({ label: value })} />
      <div className="field-grid three">
        <label className="field compact-field">
          <span>{t.draw}</span>
          <select value={plot.render_mode} onChange={(event) => onChange({ render_mode: event.target.value as DataPlotMode })}>
            {dataPlotModes.map((mode) => <option key={mode} value={mode}>{t.drawMode[mode]}</option>)}
          </select>
        </label>
        <ColumnSelect label={t.x} value={plot.x_col_index} sheet={sheet} onChange={(value) => onChange({ x_col_index: value })} t={t} />
        <ColumnSelect label={t.y} value={plot.y_col_index} sheet={sheet} onChange={(value) => onChange({ y_col_index: value })} t={t} />
      </div>
      <div className="field-grid two">
        <NumberInput label={t.startRow} value={plot.row_start} placeholder="0" onChange={(value) => onChange({ row_start: value })} />
        <NumberInput label={t.endRow} value={plot.row_end} placeholder="all" onChange={(value) => onChange({ row_end: value })} />
      </div>
      {plot.render_mode === 'fit' && (
        <>
          <label className="field compact-field">
            <span>{t.fit}</span>
            <select value={plot.fit_model ?? 'linear'} onChange={(event) => onChange({ fit_model: event.target.value as FitModel })}>
              {fitModels.map((model) => <option key={model} value={model}>{t.fitModel[model]}</option>)}
            </select>
          </label>
          <div className="toggle-row">
            {(plot.fit_model ?? 'linear') === 'linear' && (
              <label><input type="checkbox" checked={Boolean(plot.force_through_origin)} onChange={(event) => onChange({ force_through_origin: event.target.checked })} /> {t.forceOrigin}</label>
            )}
            <label><input type="checkbox" checked={Boolean(plot.start_at_zero)} onChange={(event) => onChange({ start_at_zero: event.target.checked })} /> {t.startAtZero}</label>
            <label><input type="checkbox" checked={Boolean(plot.extrapolate)} onChange={(event) => onChange({ extrapolate: event.target.checked })} /> {t.extrapolate}</label>
            <label><input type="checkbox" checked={plot.show_points ?? true} onChange={(event) => onChange({ show_points: event.target.checked })} /> {t.points}</label>
          </div>
          {(plot.fit_model ?? 'linear') === 'polynomial' && (
            <NumberInput label={t.degree} value={plot.polynomial_degree ?? 2} onChange={(value) => onChange({ polynomial_degree: Math.max(1, Math.min(8, Math.trunc(value ?? 2))) })} />
          )}
          <div className="subsection-label">{t.stats}</div>
          <div className="toggle-row compact-toggle-row">
            <label><input type="checkbox" checked={plot.show_stats ?? true} onChange={(event) => onChange({ show_stats: event.target.checked })} /> {t.show}</label>
            {(plot.fit_model ?? 'linear') === 'gaussian' ? (
              <>
                <label><input type="checkbox" checked={plot.show_gaussian_baseline ?? true} onChange={(event) => onChange({ show_gaussian_baseline: event.target.checked })} /> {t.baseline}</label>
                <label><input type="checkbox" checked={plot.show_gaussian_amplitude ?? true} onChange={(event) => onChange({ show_gaussian_amplitude: event.target.checked })} /> {t.amplitude}</label>
                <label><input type="checkbox" checked={plot.show_gaussian_mean ?? true} onChange={(event) => onChange({ show_gaussian_mean: event.target.checked })} /> {t.mean}</label>
                <label><input type="checkbox" checked={plot.show_gaussian_std ?? true} onChange={(event) => onChange({ show_gaussian_std: event.target.checked })} /> {t.std}</label>
              </>
            ) : (plot.fit_model ?? 'linear') === 'polynomial' ? (
              <label><input type="checkbox" checked={plot.show_polynomial_coefficients ?? true} onChange={(event) => onChange({ show_polynomial_coefficients: event.target.checked })} /> {t.coefficients}</label>
            ) : (plot.fit_model ?? 'linear') === 'exponential' ? (
              <>
                <label><input type="checkbox" checked={plot.show_exponential_coefficient ?? true} onChange={(event) => onChange({ show_exponential_coefficient: event.target.checked })} /> {t.coefficient}</label>
                <label><input type="checkbox" checked={plot.show_exponential_rate ?? true} onChange={(event) => onChange({ show_exponential_rate: event.target.checked })} /> {t.rate}</label>
              </>
            ) : (
              <>
                <label><input type="checkbox" checked={plot.show_slope ?? true} onChange={(event) => onChange({ show_slope: event.target.checked })} /> {t.slope}</label>
                <label><input type="checkbox" checked={plot.show_intercept ?? true} onChange={(event) => onChange({ show_intercept: event.target.checked })} /> {t.intercept}</label>
              </>
            )}
            <label><input type="checkbox" checked={plot.show_r_squared ?? true} onChange={(event) => onChange({ show_r_squared: event.target.checked })} /> R^2</label>
          </div>
        </>
      )}
      {plot.render_mode === 'steps' && (
        <label className="field compact-field">
          <span>{t.step}</span>
          <select value={plot.step_where ?? 'post'} onChange={(event) => onChange({ step_where: event.target.value as WorkbookPlotConfig['step_where'] })}>
            {stepOptions.map((option) => <option key={option} value={option}>{t.stepOption[option]}</option>)}
          </select>
        </label>
      )}
      <div className="field-grid three">
        <label className="field compact-field">
          <span>{t.color}</span>
          <input type="color" value={plot.color || '#2563eb'} onChange={(event) => onChange({ color: event.target.value })} />
        </label>
        <label className="field compact-field">
          <span>{t.line}</span>
          <select value={plot.linestyle ?? '-'} onChange={(event) => onChange({ linestyle: event.target.value })}>
            <option value="-">{t.lineStyle.solid}</option>
            <option value="--">{t.lineStyle.dash}</option>
            <option value=":">{t.lineStyle.dot}</option>
            <option value="-.">{t.lineStyle.dashDot}</option>
          </select>
        </label>
        <label className="field compact-field">
          <span>{t.marker}</span>
          <select value={plot.marker ?? 'o'} onChange={(event) => onChange({ marker: event.target.value === 'none' ? null : event.target.value })}>
            <option value="o">{t.markerStyle.circle}</option>
            <option value="s">{t.markerStyle.square}</option>
            <option value="^">{t.markerStyle.triangle}</option>
            <option value="x">{t.markerStyle.cross}</option>
            <option value="none">{t.markerStyle.none}</option>
          </select>
        </label>
      </div>
    </>
  )
}

function FormulaPlotFields({ plot, onChange, t }: { plot: FormulaPlotConfig; onChange: (patch: Partial<FormulaPlotConfig>) => void; t: AppTranslations }) {
  return (
    <>
      <TextInput label={t.label} value={plot.label} onChange={(value) => onChange({ label: value })} />
      <TextInput label={t.expression} value={plot.expression} placeholder="exp(-x^2)" onChange={(value) => onChange({ expression: value })} />
      <TextInput label={t.latex} value={plot.equation_latex} placeholder="y=e^{-x^2}" onChange={(value) => onChange({ equation_latex: value })} />
      <div className="field-grid three">
        <NumberInput label={t.min} value={plot.x_min} onChange={(value) => onChange({ x_min: value })} />
        <NumberInput label={t.max} value={plot.x_max} onChange={(value) => onChange({ x_max: value })} />
        <NumberInput label={t.points} value={plot.num_points ?? 400} onChange={(value) => onChange({ num_points: value ?? 400 })} />
      </div>
      <div className="field-grid three">
        <label className="field compact-field">
          <span>{t.color}</span>
          <input type="color" value={plot.color || '#9333ea'} onChange={(event) => onChange({ color: event.target.value })} />
        </label>
        <label className="field compact-field">
          <span>{t.line}</span>
          <select value={plot.linestyle ?? '--'} onChange={(event) => onChange({ linestyle: event.target.value })}>
            <option value="-">{t.lineStyle.solid}</option>
            <option value="--">{t.lineStyle.dash}</option>
            <option value=":">{t.lineStyle.dot}</option>
            <option value="-.">{t.lineStyle.dashDot}</option>
          </select>
        </label>
        <NumberInput label={t.width} value={plot.linewidth ?? 2} onChange={(value) => onChange({ linewidth: value ?? 2 })} />
      </div>
    </>
  )
}

function PlotsEditor({
  plots,
  sheet,
  onPatch,
  onReplace,
  onAddWorkbook,
  onAddFormula,
  onRemove,
  t
}: {
  plots: PlotItemConfig[]
  sheet: WorksheetData | null
  onPatch: (index: number, patch: Partial<PlotItemConfig>) => void
  onReplace: (index: number, plot: PlotItemConfig) => void
  onAddWorkbook: () => void
  onAddFormula: () => void
  onRemove: (index: number) => void
  t: AppTranslations
}) {
  const [collapsedPlots, setCollapsedPlots] = useState<Record<string, boolean>>({})

  function togglePlot(plotId: string) {
    setCollapsedPlots((current) => ({ ...current, [plotId]: !current[plotId] }))
  }

  return (
    <section className="panel-section">
      <div className="section-title-row">
        <h2>{t.plots}</h2>
        <PlotActions onAddWorkbook={onAddWorkbook} onAddFormula={onAddFormula} t={t} />
      </div>
      <div className="stack">
        {plots.map((plot, index) => {
          const isCollapsed = Boolean(collapsedPlots[plot.id])
          return (
            <div className={`item-card${isCollapsed ? ' collapsed' : ''}`} key={plot.id}>
              <div className="item-header">
                <button className="item-title-button" type="button" onClick={() => togglePlot(plot.id)} aria-expanded={!isCollapsed}>
                  {isCollapsed ? <ChevronRight size={15} /> : <ChevronDown size={15} />}
                  <span>{t.plot(index + 1)}</span>
                  <span className="item-title-meta">{plot.label || (plot.source_type === 'formula' ? t.formulaItem(index + 1) : t.plot(index + 1))}</span>
                </button>
                <button className="icon-button danger" type="button" onClick={() => onRemove(index)} title={t.remove}><Trash2 size={15} /></button>
              </div>
              {!isCollapsed && (
                <div className="item-body">
                  <label className="field compact-field">
                    <span>{t.source}</span>
                    <select
                      value={plot.source_type}
                      onChange={(event) => {
                        const nextType = event.target.value
                        onReplace(index, nextType === 'formula' ? createFormulaPlot(index, t, plot.label || t.formulaItem(index + 1)) : createWorkbookPlot(index, t, plot.label || t.plot(index + 1)))
                      }}
                    >
                      <option value="workbook">{t.sourceOption.workbook}</option>
                      <option value="formula">{t.sourceOption.formula}</option>
                    </select>
                  </label>
                  {plot.source_type === 'formula' ? (
                    <FormulaPlotFields plot={plot} onChange={(patch) => onPatch(index, patch as Partial<PlotItemConfig>)} t={t} />
                  ) : (
                    <WorkbookPlotFields plot={plot} sheet={sheet} onChange={(patch) => onPatch(index, patch as Partial<PlotItemConfig>)} t={t} />
                  )}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </section>
  )
}

function LegendInfoEditor({
  items,
  onAdd,
  onPatch,
  onRemove,
  t
}: {
  items: LegendInfoItem[]
  onAdd: () => void
  onPatch: (index: number, patch: Partial<LegendInfoItem>) => void
  onRemove: (index: number) => void
  t: AppTranslations
}) {
  return (
    <section className="panel-section">
      <div className="section-title-row">
        <h2>{t.legendInfo}</h2>
        <button className="secondary-button" type="button" onClick={onAdd}><Plus size={15} /><span>{t.info}</span></button>
      </div>
      <div className="stack">
        {items.length === 0 && <div className="muted-note">{t.legendInfoEmpty}</div>}
        {items.map((item, index) => (
          <div className="item-card" key={item.id}>
            <div className="item-header">
              <span>{t.infoItem(index + 1)}</span>
              <button className="icon-button danger" type="button" onClick={() => onRemove(index)} title={t.remove}><Trash2 size={15} /></button>
            </div>
            <TextInput label={t.label} value={item.label} placeholder="m_1 = 1kg" onChange={(value) => onPatch(index, { label: value })} />
          </div>
        ))}
      </div>
    </section>
  )
}

function WorkbookEditor({
  sheet,
  onColumnNameChange,
  onCellChange,
  onAddRow,
  onAddColumn,
  t
}: {
  sheet: WorksheetData | null
  onColumnNameChange: (columnIndex: number, name: string) => void
  onCellChange: (rowIndex: number, columnIndex: number, value: string) => void
  onAddRow: () => void
  onAddColumn: () => void
  t: AppTranslations
}) {
  if (!sheet) {
    return <section className="data-editor empty-editor">{t.uploadFileToEditData}</section>
  }

  return (
    <section className="data-editor">
      <div className="data-editor-header">
        <h2>{t.fileData}</h2>
        <div className="data-editor-actions">
          <button className="secondary-button" type="button" onClick={onAddRow}><Plus size={15} /><span>{t.row}</span></button>
          <button className="secondary-button" type="button" onClick={onAddColumn}><Plus size={15} /><span>{t.column}</span></button>
        </div>
      </div>
      <div className="table-scroll">
        <table className="workbook-table">
          <thead>
            <tr>
              <th className="row-index">#</th>
              {sheet.columns.map((column) => (
                <th key={column.index}>
                  <input
                    aria-label={t.columnNameAria(column.index + 1)}
                    value={column.name}
                    onChange={(event) => onColumnNameChange(column.index, event.target.value)}
                  />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sheet.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                <th className="row-index">{rowIndex + 1}</th>
                {sheet.columns.map((column) => (
                  <td key={column.index}>
                    <input
                      aria-label={t.cellAria(rowIndex + 1, column.index + 1)}
                      value={formatEditableCell(row[column.index])}
                      onChange={(event) => onCellChange(rowIndex, column.index, event.target.value)}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}

function CollapsedPaneRail({ label, title, direction, onExpand }: { label: string; title: string; direction: 'left' | 'right'; onExpand: () => void }) {
  return (
    <button className="collapsed-pane-rail" type="button" title={title} onClick={onExpand}>
      {direction === 'left' ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      <span>{label}</span>
    </button>
  )
}

function ParametersPanel({
  summaries,
  curveEquations,
  t,
  onCollapse
}: {
  summaries: string[]
  curveEquations: Array<{ id: string; label: string; equation: string }>
  t: AppTranslations
  onCollapse: () => void
}) {
  return (
    <aside className="parameters-panel">
      <header className="parameters-header">
        <div>
          <h2>{t.parameters}</h2>
          <p>{t.noEquationsYet}</p>
        </div>
        <button className="icon-button" type="button" title={t.collapsePanel} onClick={onCollapse}><ChevronRight size={16} /></button>
      </header>

      {summaries.length > 0 && (
        <section className="parameters-section">
          <h3>{t.fitEquations}</h3>
          <div className="equation-list">
            {summaries.map((summary) => (
              <div className="equation-card" key={summary}>{summary}</div>
            ))}
          </div>
        </section>
      )}

      {curveEquations.length > 0 && (
        <section className="parameters-section">
          <h3>{t.curveEquations}</h3>
          <div className="equation-list">
            {curveEquations.map((entry) => (
              <div className="equation-card" key={entry.id}>
                <strong>{entry.label}</strong>
                <span>{entry.equation}</span>
              </div>
            ))}
          </div>
        </section>
      )}

      {summaries.length === 0 && curveEquations.length === 0 && (
        <div className="parameters-empty">{t.noEquationsYet}</div>
      )}
    </aside>
  )
}

function App() {
  const [language, setLanguage] = useState<AppLanguage>(() => detectInitialLanguage())
  const t = useMemo(() => getTranslations(language), [language])
  const shellRef = useRef<HTMLElement>(null)
  const [workbook, setWorkbook] = useState<WorkbookData | null>(null)
  const [sheetName, setSheetName] = useState('')
  const [config, setConfig] = useState<EditablePlotConfig>(() => createEmptyConfig(getTranslations(detectInitialLanguage())))
  const [uploadError, setUploadError] = useState('')
  const [templateMessage, setTemplateMessage] = useState('')
  const [templateError, setTemplateError] = useState('')
  const [plotPaneWeight, setPlotPaneWeight] = useState(64)
  const [leftPaneWeight, setLeftPaneWeight] = useState(24)
  const [rightPaneWeight, setRightPaneWeight] = useState(22)
  const [collapsedPanes, setCollapsedPanes] = useState({ left: false, middle: false, right: false })
  const plotRef = useRef<PlotPreviewHandle>(null)
  const templateInputRef = useRef<HTMLInputElement>(null)
  const workspaceBodyRef = useRef<HTMLDivElement>(null)
  const activeSheet = useMemo(() => getSheet(workbook, sheetName), [workbook, sheetName])
  const figure = useMemo(() => buildPlotFigure(config, activeSheet, language), [config, activeSheet, language])

  function resizeWorkspace(clientY: number) {
    const container = workspaceBodyRef.current
    if (!container) {
      return
    }
    const bounds = container.getBoundingClientRect()
    const nextWeight = ((clientY - bounds.top) / bounds.height) * 100
    setPlotPaneWeight(Math.min(80, Math.max(30, nextWeight)))
  }

  function startWorkspaceResize(clientY: number) {
    resizeWorkspace(clientY)
    document.body.style.cursor = 'row-resize'
    document.body.style.userSelect = 'none'

    const resizeFromMouse = (moveEvent: MouseEvent) => resizeWorkspace(moveEvent.clientY)
    const resizeFromTouch = (moveEvent: TouchEvent) => {
      if (moveEvent.touches[0]) {
        resizeWorkspace(moveEvent.touches[0].clientY)
      }
    }
    const stopResize = () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', resizeFromMouse)
      window.removeEventListener('mouseup', stopResize)
      window.removeEventListener('touchmove', resizeFromTouch)
      window.removeEventListener('touchend', stopResize)
    }

    window.addEventListener('mousemove', resizeFromMouse)
    window.addEventListener('mouseup', stopResize)
    window.addEventListener('touchmove', resizeFromTouch)
    window.addEventListener('touchend', stopResize)
  }

  async function handleWorkbook(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }
    setUploadError('')
    try {
      const parsed = await parseWorkbook(file)
      setWorkbook(parsed)
      setSheetName(parsed.sheets[0]?.name ?? '')
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setUploadError(localizeRuntimeMessage(message, language))
    }
  }

  function patchConfig(patch: Partial<EditablePlotConfig>) {
    setConfig((current) => ({ ...current, ...patch }))
  }

  function resetPlot() {
    setConfig(createEmptyConfig(t))
    setTemplateMessage('')
    setTemplateError('')
  }

  function downloadTemplate(format: 'json' | 'csv') {
    setTemplateError('')
    const template = buildWorkbookSettingsTemplate(config, workbook, sheetName, activeSheet)
    const baseName = sanitizeFileBaseName(template.workbook.fileName || config.name || 'plot-settings')
    if (format === 'csv') {
      downloadTextFile(`${baseName}-plot-template.csv`, 'text/csv;charset=utf-8', templateToCsv(template))
    } else {
      downloadTextFile(`${baseName}-plot-template.json`, 'application/json;charset=utf-8', `${JSON.stringify(template, null, 2)}\n`)
    }
    setTemplateMessage(t.downloadedTemplateSettings(format.toUpperCase()))
  }

  async function importTemplate(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) {
      return
    }
    setTemplateMessage('')
    setTemplateError('')
    try {
      const template = parseWorkbookSettingsTemplate(await file.text(), file.name)
      setConfig(withFreshPlotIds(template.config, t))
      if (workbook && template.workbook.sheetName && workbook.sheets.some((sheet) => sheet.name === template.workbook.sheetName)) {
        setSheetName(template.workbook.sheetName)
      }
      const templateWorkbookName = workbook && template.workbook.fileName && template.workbook.fileName !== workbook.fileName
        ? template.workbook.fileName
        : undefined
      setTemplateMessage(t.importedTemplateSettings(templateWorkbookName))
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setTemplateError(localizeRuntimeMessage(message, language))
    }
  }

  function updateActiveSheet(updater: (sheet: WorksheetData) => WorksheetData) {
    setWorkbook((current) => {
      if (!current) {
        return current
      }
      return {
        ...current,
        sheets: current.sheets.map((sheet) => sheet.name === sheetName ? updater(sheet) : sheet)
      }
    })
  }

  function patchPlot(index: number, patch: Partial<PlotItemConfig>) {
    setConfig((current) => ({
      ...current,
      plots: current.plots.map((plot, plotIndex) => plotIndex === index ? { ...plot, ...patch } as PlotItemConfig : plot)
    }))
  }

  function replacePlot(index: number, plot: PlotItemConfig) {
    setConfig((current) => ({
      ...current,
      plots: current.plots.map((entry, entryIndex) => entryIndex === index ? plot : entry)
    }))
  }

  function patchLegendInfo(index: number, patch: Partial<LegendInfoItem>) {
    setConfig((current) => ({
      ...current,
      legend_info: (current.legend_info ?? []).map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item)
    }))
  }

  const localizedFigureErrors = useMemo(
    () => figure.errors.map((message) => localizeRuntimeMessage(message, language)),
    [figure.errors, language]
  )
  const curveEquations = useMemo(
    () => config.plots.flatMap((plot, index) => {
      if (plot.source_type !== 'formula') {
        return []
      }
      const equation = formatLatexText(plot.equation_latex || plot.expression || '').trim()
      if (!equation) {
        return []
      }
      return [{ id: plot.id, label: plot.label || t.formulaItem(index + 1), equation }]
    }),
    [config.plots, t]
  )
  const middlePaneWeight = 100 - leftPaneWeight - rightPaneWeight
  const shellGridColumns = [
    collapsedPanes.left ? collapsedPaneSize : `minmax(320px, ${leftPaneWeight}fr)`,
    '12px',
    collapsedPanes.middle ? collapsedPaneSize : `minmax(0, ${middlePaneWeight}fr)`,
    '12px',
    collapsedPanes.right ? collapsedPaneSize : `minmax(260px, ${rightPaneWeight}fr)`
  ].join(' ')

  function setPaneCollapsed(pane: keyof typeof collapsedPanes, collapsed: boolean) {
    setCollapsedPanes((current) => ({ ...current, [pane]: collapsed }))
  }

  function startShellResize(kind: 'left' | 'right', clientX: number) {
    if ((kind === 'left' && collapsedPanes.left) || (kind === 'right' && collapsedPanes.right)) {
      return
    }
    const shell = shellRef.current
    if (!shell) {
      return
    }

    const resizeShell = (nextClientX: number) => {
      const bounds = shell.getBoundingClientRect()
      if (kind === 'left') {
        const nextLeft = ((nextClientX - bounds.left) / bounds.width) * 100
        const maxLeft = 100 - rightPaneWeight - minimumMiddlePaneWeight
        setLeftPaneWeight(Math.min(maxLeft, Math.max(minimumLeftPaneWeight, nextLeft)))
        return
      }

      const nextRight = ((bounds.right - nextClientX) / bounds.width) * 100
      const maxRight = 100 - leftPaneWeight - minimumMiddlePaneWeight
      setRightPaneWeight(Math.min(maxRight, Math.max(minimumRightPaneWeight, nextRight)))
    }

    resizeShell(clientX)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'

    const resizeFromMouse = (moveEvent: MouseEvent) => resizeShell(moveEvent.clientX)
    const resizeFromTouch = (moveEvent: TouchEvent) => {
      if (moveEvent.touches[0]) {
        resizeShell(moveEvent.touches[0].clientX)
      }
    }
    const stopResize = () => {
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      window.removeEventListener('mousemove', resizeFromMouse)
      window.removeEventListener('mouseup', stopResize)
      window.removeEventListener('touchmove', resizeFromTouch)
      window.removeEventListener('touchend', stopResize)
    }

    window.addEventListener('mousemove', resizeFromMouse)
    window.addEventListener('mouseup', stopResize)
    window.addEventListener('touchmove', resizeFromTouch)
    window.addEventListener('touchend', stopResize)
  }

  return (
    <main
      className="app-shell"
      ref={shellRef}
      style={{ gridTemplateColumns: shellGridColumns }}
    >
      {collapsedPanes.left ? (
        <CollapsedPaneRail label={t.settingsPanel} title={t.expandPanel} direction="right" onExpand={() => setPaneCollapsed('left', false)} />
      ) : (
        <aside className="control-panel">
          <div className="panel-toolbar">
            <label className="field compact-field language-field">
              <span>{t.languageLabel}</span>
              <select value={language} onChange={(event) => setLanguage(event.target.value as AppLanguage)}>
                {languageOptions.map((option) => (
                  <option key={option} value={option}>{option === 'en' ? t.english : t.romanian}</option>
                ))}
              </select>
            </label>
            <label className="upload-button">
              <Upload size={16} />
              <span>{t.uploadFile}</span>
              <input type="file" accept=".xlsx" onChange={handleWorkbook} />
            </label>
            <button className="icon-button" type="button" title={t.resetPlot} onClick={resetPlot}><RefreshCw size={16} /></button>
            <button className="icon-button" type="button" title={t.collapsePanel} onClick={() => setPaneCollapsed('left', true)}><ChevronLeft size={16} /></button>
          </div>

          {uploadError && <div className="alert error">{uploadError}</div>}
          {templateError && <div className="alert error">{templateError}</div>}
          {templateMessage && <div className="alert info">{templateMessage}</div>}

          <section className="panel-section">
            <h2>{t.source}</h2>
            <label className="field">
              <span>{t.file}</span>
              <input readOnly value={workbook?.fileName ?? ''} placeholder={t.noFileSelected} />
            </label>
            <label className="field">
              <span>{t.sheet}</span>
              <select value={sheetName} onChange={(event) => setSheetName(event.target.value)} disabled={!workbook}>
                {workbook?.sheets.map((sheet) => <option key={sheet.name} value={sheet.name}>{sheet.name}</option>)}
              </select>
            </label>
            <div className="template-actions">
              <button className="secondary-button" type="button" onClick={() => downloadTemplate('json')} disabled={!workbook}>
                <Download size={15} />
                <span>{t.templateJson}</span>
              </button>
              <button className="secondary-button" type="button" onClick={() => downloadTemplate('csv')} disabled={!workbook}>
                <Download size={15} />
                <span>{t.templateCsv}</span>
              </button>
              <button className="secondary-button" type="button" onClick={() => templateInputRef.current?.click()}>
                <Upload size={15} />
                <span>{t.import}</span>
              </button>
              <input ref={templateInputRef} className="hidden-file-input" type="file" accept=".json,.csv,application/json,text/csv" onChange={importTemplate} />
            </div>
          </section>

          <PlotsEditor
            plots={config.plots}
            sheet={activeSheet}
            onPatch={patchPlot}
            onReplace={replacePlot}
            onAddWorkbook={() => setConfig((current) => ({ ...current, plots: [...current.plots, createWorkbookPlot(current.plots.length, t)] }))}
            onAddFormula={() => setConfig((current) => ({ ...current, plots: [...current.plots, createFormulaPlot(current.plots.length, t)] }))}
            onRemove={(index) => setConfig((current) => ({ ...current, plots: current.plots.filter((_plot, plotIndex) => plotIndex !== index) }))}
            t={t}
          />

          <LegendInfoEditor
            items={config.legend_info ?? []}
            onAdd={() => setConfig((current) => ({ ...current, legend_info: [...(current.legend_info ?? []), createLegendInfoItem(current.legend_info?.length ?? 0, t)] }))}
            onPatch={patchLegendInfo}
            onRemove={(index) => setConfig((current) => ({ ...current, legend_info: (current.legend_info ?? []).filter((_item, itemIndex) => itemIndex !== index) }))}
            t={t}
          />

          <section className="panel-section">
            <h2>{t.axes}</h2>
            <TextInput label={t.title} value={config.plot_name} onChange={(value) => patchConfig({ plot_name: value })} />
            <div className="field-grid two">
              <TextInput label={t.xLabel} value={config.x_label} onChange={(value) => patchConfig({ x_label: value })} />
              <TextInput label={t.yLabel} value={config.y_label} onChange={(value) => patchConfig({ y_label: value })} />
              <TextInput label={t.xUnit} value={config.x_unit} onChange={(value) => patchConfig({ x_unit: value })} />
              <TextInput label={t.yUnit} value={config.y_unit} onChange={(value) => patchConfig({ y_unit: value })} />
              <NumberInput label={t.xExp} value={config.x_exponent} onChange={(value) => patchConfig({ x_exponent: value ?? 0 })} />
              <NumberInput label={t.yExp} value={config.y_exponent} onChange={(value) => patchConfig({ y_exponent: value ?? 0 })} />
              <label className="field compact-field">
                <span>{t.stats}</span>
                <select value={config.stats_pos} onChange={(event) => patchConfig({ stats_pos: event.target.value as EditablePlotConfig['stats_pos'] })}>
                  <option value="top-right">{t.statsPositionOption.topRight}</option>
                  <option value="top-left">{t.statsPositionOption.topLeft}</option>
                  <option value="bottom-right">{t.statsPositionOption.bottomRight}</option>
                  <option value="bottom-left">{t.statsPositionOption.bottomLeft}</option>
                </select>
              </label>
            </div>
            <div className="toggle-row">
              <label><input type="checkbox" checked={Boolean(config.x_start_at_zero)} onChange={(event) => patchConfig({ x_start_at_zero: event.target.checked })} /> {t.xStartsZero}</label>
              <label><input type="checkbox" checked={Boolean(config.y_start_at_zero)} onChange={(event) => patchConfig({ y_start_at_zero: event.target.checked })} /> {t.yStartsZero}</label>
            </div>
          </section>
        </aside>
      )}

      <div
        className="shell-resize-handle"
        role="separator"
        aria-label={t.resizeSettingsPanel}
        aria-orientation="vertical"
        aria-valuemin={minimumLeftPaneWeight}
        aria-valuemax={100 - rightPaneWeight - minimumMiddlePaneWeight}
        aria-valuenow={Math.round(leftPaneWeight)}
        tabIndex={0}
        title={t.resizeSettingsPanel}
        onMouseDown={(event) => {
          event.preventDefault()
          startShellResize('left', event.clientX)
        }}
        onTouchStart={(event) => {
          const touch = event.touches[0]
          if (touch) {
            startShellResize('left', touch.clientX)
          }
        }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowLeft') {
            event.preventDefault()
            setLeftPaneWeight((current) => Math.max(minimumLeftPaneWeight, current - 2))
          }
          if (event.key === 'ArrowRight') {
            event.preventDefault()
            setLeftPaneWeight((current) => Math.min(100 - rightPaneWeight - minimumMiddlePaneWeight, current + 2))
          }
        }}
      >
        <GripVertical size={18} />
      </div>

      {collapsedPanes.middle ? (
        <CollapsedPaneRail label={t.plotterPanel} title={t.expandPanel} direction="right" onExpand={() => setPaneCollapsed('middle', false)} />
      ) : (
        <section className="workspace-panel">
          <header className="workspace-header">
            <p>{activeSheet ? t.rowsColumns(activeSheet.rows.length, activeSheet.columns.length) : t.noUploadedFileInMemory}</p>
            <div className="workspace-actions">
              <button className="primary-button" type="button" onClick={() => plotRef.current?.downloadPng()} disabled={figure.data.length === 0}>
                <Download size={17} />
                <span>{t.png}</span>
              </button>
              <button className="icon-button" type="button" title={t.collapsePanel} onClick={() => setPaneCollapsed('middle', true)}><ChevronLeft size={16} /></button>
            </div>
          </header>

          {localizedFigureErrors.length > 0 && <div className="alert error">{localizedFigureErrors.join(' ')}</div>}
          <div
            className="workspace-body"
            ref={workspaceBodyRef}
            style={{ gridTemplateRows: `minmax(240px, ${plotPaneWeight}fr) 12px minmax(160px, ${100 - plotPaneWeight}fr)` }}
          >
            <PlotPreview
              ref={plotRef}
              figure={figure}
              exportLabels={{
                title: t.pngPreview,
                help: t.pngPreviewHelp,
                frame: t.exportFrame,
                frameHelp: t.exportFrameHelp,
                download: t.downloadPng,
                cancel: t.cancel
              }}
            />
            <div
              className="workspace-resize-handle"
              role="separator"
              aria-label={t.resizePanels}
              aria-orientation="horizontal"
              aria-valuemin={30}
              aria-valuemax={80}
              aria-valuenow={Math.round(plotPaneWeight)}
              tabIndex={0}
              title={t.resizePanels}
              onMouseDown={(event) => {
                event.preventDefault()
                startWorkspaceResize(event.clientY)
              }}
              onTouchStart={(event) => {
                const touch = event.touches[0]
                if (touch) {
                  startWorkspaceResize(touch.clientY)
                }
              }}
              onKeyDown={(event) => {
                if (event.key === 'ArrowUp') {
                  event.preventDefault()
                  setPlotPaneWeight((current) => Math.max(30, current - 4))
                }
                if (event.key === 'ArrowDown') {
                  event.preventDefault()
                  setPlotPaneWeight((current) => Math.min(80, current + 4))
                }
              }}
            >
              <GripHorizontal size={18} />
            </div>
            <WorkbookEditor
              sheet={activeSheet}
              onColumnNameChange={(columnIndex, name) => updateActiveSheet((sheet) => updateColumnName(sheet, columnIndex, name))}
              onCellChange={(rowIndex, columnIndex, value) => updateActiveSheet((sheet) => updateCell(sheet, rowIndex, columnIndex, parseEditableCell(value)))}
              onAddRow={() => updateActiveSheet(addSheetRow)}
              onAddColumn={() => updateActiveSheet(addSheetColumn)}
              t={t}
            />
          </div>
        </section>
      )}

      <div
        className="shell-resize-handle"
        role="separator"
        aria-label={t.resizeParametersPanel}
        aria-orientation="vertical"
        aria-valuemin={minimumRightPaneWeight}
        aria-valuemax={100 - leftPaneWeight - minimumMiddlePaneWeight}
        aria-valuenow={Math.round(rightPaneWeight)}
        tabIndex={0}
        title={t.resizeParametersPanel}
        onMouseDown={(event) => {
          event.preventDefault()
          startShellResize('right', event.clientX)
        }}
        onTouchStart={(event) => {
          const touch = event.touches[0]
          if (touch) {
            startShellResize('right', touch.clientX)
          }
        }}
        onKeyDown={(event) => {
          if (event.key === 'ArrowLeft') {
            event.preventDefault()
            setRightPaneWeight((current) => Math.min(100 - leftPaneWeight - minimumMiddlePaneWeight, current + 2))
          }
          if (event.key === 'ArrowRight') {
            event.preventDefault()
            setRightPaneWeight((current) => Math.max(minimumRightPaneWeight, current - 2))
          }
        }}
      >
        <GripVertical size={18} />
      </div>

      {collapsedPanes.right ? (
        <CollapsedPaneRail label={t.parameters} title={t.expandPanel} direction="left" onExpand={() => setPaneCollapsed('right', false)} />
      ) : (
        <ParametersPanel summaries={figure.summaries} curveEquations={curveEquations} t={t} onCollapse={() => setPaneCollapsed('right', true)} />
      )}
    </main>
  )
}

export default App
