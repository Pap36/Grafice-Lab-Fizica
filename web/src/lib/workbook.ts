import readXlsxFile, { type Sheet } from 'read-excel-file/browser'
import type { WorkbookColumn, WorkbookData, WorksheetData } from '../types/templates'

function cellToString(value: unknown, fallback: string): string {
  if (value === null || value === undefined || value === '') {
    return fallback
  }
  return String(value)
}

export function columnName(index: number): string {
  let remaining = index + 1
  let name = ''
  while (remaining > 0) {
    const modulo = (remaining - 1) % 26
    name = String.fromCharCode(65 + modulo) + name
    remaining = Math.floor((remaining - modulo) / 26)
  }
  return name
}

export function formatEditableCell(value: unknown): string {
  if (value === null || value === undefined) {
    return ''
  }
  return String(value)
}

export function parseEditableCell(value: string): unknown {
  const trimmed = value.trim()
  if (trimmed === '') {
    return ''
  }
  if (/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(trimmed)) {
    return Number(trimmed)
  }
  return value
}

function normalizeRowLength(row: unknown[], columnCount: number): unknown[] {
  return Array.from({ length: columnCount }, (_unused, index) => row[index] ?? '')
}

export function updateColumnName(sheet: WorksheetData, columnIndex: number, name: string): WorksheetData {
  return {
    ...sheet,
    columns: sheet.columns.map((column) => column.index === columnIndex ? { ...column, name } : column)
  }
}

export function updateCell(sheet: WorksheetData, rowIndex: number, columnIndex: number, value: unknown): WorksheetData {
  return {
    ...sheet,
    rows: sheet.rows.map((row, index) => {
      if (index !== rowIndex) {
        return row
      }
      const nextRow = normalizeRowLength(row, sheet.columns.length)
      nextRow[columnIndex] = value
      return nextRow
    })
  }
}

export function addSheetRow(sheet: WorksheetData): WorksheetData {
  return {
    ...sheet,
    rows: [...sheet.rows, Array.from({ length: sheet.columns.length }, () => '')]
  }
}

export function addSheetColumn(sheet: WorksheetData): WorksheetData {
  const nextIndex = sheet.columns.length
  return {
    ...sheet,
    columns: [...sheet.columns, { index: nextIndex, name: columnName(nextIndex) }],
    rows: sheet.rows.map((row) => [...normalizeRowLength(row, nextIndex), ''])
  }
}

function normalizeSheet(name: string, matrix: unknown[][]): WorksheetData {
  const header = matrix[0] ?? []
  const maxColumns = Math.max(
    header.length,
    ...matrix.slice(1).map((row) => row.length),
    0
  )
  const columns: WorkbookColumn[] = Array.from({ length: maxColumns }, (_unused, index) => ({
    index,
    name: cellToString(header[index], columnName(index))
  }))

  return {
    name,
    columns,
    rows: matrix.slice(1)
  }
}

export async function parseWorkbook(file: File): Promise<WorkbookData> {
  if (!file.name.toLowerCase().endsWith('.xlsx')) {
    throw new Error('Please upload an .xlsx file.')
  }
  const workbookSheets = await readXlsxFile(file) as Sheet[]
  const sheets = workbookSheets
    .map((sheet) => normalizeSheet(sheet.sheet, sheet.data as unknown[][]))
    .filter((sheet) => sheet.columns.length > 0)

  if (sheets.length === 0) {
    throw new Error('The file does not contain any readable sheets.')
  }

  return {
    fileName: file.name,
    sheets
  }
}

export function getSheet(workbook: WorkbookData | null, sheetName: string): WorksheetData | null {
  return workbook?.sheets.find((sheet) => sheet.name === sheetName) ?? null
}
