import { describe, expect, it } from 'vitest'
import { addSheetColumn, addSheetRow, parseEditableCell, updateCell, updateColumnName } from '../lib/workbook'
import type { WorksheetData } from '../types/templates'

const sheet: WorksheetData = {
  name: 'Sheet1',
  columns: [
    { index: 0, name: 'x' },
    { index: 1, name: 'y' }
  ],
  rows: [
    [1, 2],
    [3, 4]
  ]
}

describe('workbook editing helpers', () => {
  it('parses numeric cell edits and preserves text', () => {
    expect(parseEditableCell(' 3.5e2 ')).toBe(350)
    expect(parseEditableCell('label')).toBe('label')
    expect(parseEditableCell('')).toBe('')
  })

  it('updates headers, cells, rows, and columns immutably', () => {
    const renamed = updateColumnName(sheet, 1, 'velocity')
    expect(renamed.columns[1].name).toBe('velocity')
    expect(sheet.columns[1].name).toBe('y')

    const edited = updateCell(sheet, 0, 1, 12)
    expect(edited.rows[0][1]).toBe(12)
    expect(sheet.rows[0][1]).toBe(2)

    const withRow = addSheetRow(sheet)
    expect(withRow.rows).toHaveLength(3)
    expect(withRow.rows[2]).toEqual(['', ''])

    const withColumn = addSheetColumn(sheet)
    expect(withColumn.columns[2]).toMatchObject({ index: 2, name: 'C' })
    expect(withColumn.rows[0]).toEqual([1, 2, ''])
  })
})
