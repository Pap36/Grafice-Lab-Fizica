import type { EditablePlotConfig, WorkbookData, WorksheetData } from '../types/templates'

export const WORKBOOK_TEMPLATE_TYPE = 'grafice-lab-fizica.web-workbook-template'

export interface WorkbookSettingsTemplate {
  type: typeof WORKBOOK_TEMPLATE_TYPE
  version: 1
  exported_at: string
  workbook: {
    fileName: string
    sheetName: string
    columns: string[]
  }
  config: EditablePlotConfig
}

function csvEscape(value: unknown): string {
  const text = String(value ?? '')
  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

function parseCsvRows(input: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false

  for (let index = 0; index < input.length; index += 1) {
    const character = input[index]
    const nextCharacter = input[index + 1]

    if (quoted) {
      if (character === '"' && nextCharacter === '"') {
        field += '"'
        index += 1
      } else if (character === '"') {
        quoted = false
      } else {
        field += character
      }
      continue
    }

    if (character === '"') {
      quoted = true
    } else if (character === ',') {
      row.push(field)
      field = ''
    } else if (character === '\n') {
      row.push(field)
      rows.push(row)
      row = []
      field = ''
    } else if (character !== '\r') {
      field += character
    }
  }

  row.push(field)
  if (row.some((value) => value !== '') || rows.length === 0) {
    rows.push(row)
  }

  return rows
}

export function buildWorkbookSettingsTemplate(
  config: EditablePlotConfig,
  workbook: WorkbookData | null,
  sheetName: string,
  sheet: WorksheetData | null
): WorkbookSettingsTemplate {
  return {
    type: WORKBOOK_TEMPLATE_TYPE,
    version: 1,
    exported_at: new Date().toISOString(),
    workbook: {
      fileName: workbook?.fileName ?? '',
      sheetName,
      columns: sheet?.columns.map((column) => column.name) ?? []
    },
    config: JSON.parse(JSON.stringify(config)) as EditablePlotConfig
  }
}

export function templateToCsv(template: WorkbookSettingsTemplate): string {
  const headers = ['type', 'version', 'exported_at', 'workbook_file', 'sheet_name', 'columns_json', 'config_json']
  const values = [
    template.type,
    template.version,
    template.exported_at,
    template.workbook.fileName,
    template.workbook.sheetName,
    JSON.stringify(template.workbook.columns),
    JSON.stringify(template.config)
  ]
  return `${headers.join(',')}\n${values.map(csvEscape).join(',')}\n`
}

export function parseWorkbookSettingsTemplate(input: string, fileName = 'template'): WorkbookSettingsTemplate {
  const trimmed = input.trim()
  if (!trimmed) {
    throw new Error('Template file is empty.')
  }

  if (fileName.toLowerCase().endsWith('.csv')) {
    const [headers, values] = parseCsvRows(trimmed)
    const record = Object.fromEntries(headers.map((header, index) => [header, values?.[index] ?? '']))
    const config = JSON.parse(record.config_json || '{}') as EditablePlotConfig
    return assertWorkbookSettingsTemplate({
      type: record.type,
      version: Number(record.version),
      exported_at: record.exported_at,
      workbook: {
        fileName: record.workbook_file || '',
        sheetName: record.sheet_name || '',
        columns: JSON.parse(record.columns_json || '[]') as string[]
      },
      config
    })
  }

  return assertWorkbookSettingsTemplate(JSON.parse(trimmed))
}

export function assertWorkbookSettingsTemplate(value: unknown): WorkbookSettingsTemplate {
  if (!value || typeof value !== 'object') {
    throw new Error('Template must be an object.')
  }
  const template = value as WorkbookSettingsTemplate
  if (template.type !== WORKBOOK_TEMPLATE_TYPE || template.version !== 1) {
    throw new Error('Unsupported template format.')
  }
  if (!template.config || !Array.isArray(template.config.plots)) {
    throw new Error('Template does not contain interactive plot settings.')
  }
  return template
}
