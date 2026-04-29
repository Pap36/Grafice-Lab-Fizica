import { describe, expect, it } from 'vitest'
import { buildWorkbookSettingsTemplate, parseWorkbookSettingsTemplate, templateToCsv, WORKBOOK_TEMPLATE_TYPE } from '../lib/templateTransfer'
import type { EditablePlotConfig, WorkbookData, WorksheetData } from '../types/templates'

const config: EditablePlotConfig = {
  name: 'Interactive plot',
  plot_name: 'Torsiune',
  x_label: 'l',
  y_label: '\\Phi',
  plots: [
    {
      id: 'plot-1',
      source_type: 'workbook',
      render_mode: 'fit',
      fit_model: 'linear',
      label: 'm_1',
      x_col_index: 0,
      y_col_index: 1
    }
  ],
  series: [],
  fits: [],
  formula_curves: [],
  legend_info: [
    {
      id: 'info-1',
      label: 'calibrare: \\Phi_0'
    }
  ]
}

const workbook: WorkbookData = {
  fileName: 'Torsiunea Tijei.xlsx',
  sheets: []
}

const sheet: WorksheetData = {
  name: 'Sheet1',
  columns: [
    { index: 0, name: 'l' },
    { index: 1, name: 'fi' }
  ],
  rows: [[1, 2]]
}

describe('template transfer', () => {
  it('round-trips workbook settings as json-compatible data', () => {
    const template = buildWorkbookSettingsTemplate(config, workbook, sheet.name, sheet)
    const parsed = parseWorkbookSettingsTemplate(JSON.stringify(template), 'settings.json')

    expect(parsed.type).toBe(WORKBOOK_TEMPLATE_TYPE)
    expect(parsed.workbook.fileName).toBe('Torsiunea Tijei.xlsx')
    expect(parsed.workbook.columns).toEqual(['l', 'fi'])
    expect(parsed.config.plots[0].label).toBe('m_1')
    expect(parsed.config.legend_info[0].label).toBe('calibrare: \\Phi_0')
  })

  it('round-trips workbook settings through csv', () => {
    const template = buildWorkbookSettingsTemplate(config, workbook, sheet.name, sheet)
    const parsed = parseWorkbookSettingsTemplate(templateToCsv(template), 'settings.csv')

    expect(parsed.config.plot_name).toBe('Torsiune')
    expect(parsed.config.y_label).toBe('\\Phi')
    expect(parsed.config.legend_info[0].label).toBe('calibrare: \\Phi_0')
  })
})
