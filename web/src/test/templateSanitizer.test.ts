import { describe, expect, it } from 'vitest'
import { assertPublicTemplates, sanitizeTemplates } from '../lib/templateSanitizer'

const source = {
  templates: [
    {
      name: 'Private template',
      excel_path: 'Lab/data.xlsx',
      output: 'Lab/plot.png',
      x_label: 'x',
      series: [
        {
          label: 'series',
          excel_path: 'Lab/series.xls',
          x_col_index: 0,
          y_col_index: 1
        }
      ]
    }
  ]
}

describe('template sanitizer', () => {
  it('removes private path and artifact fields recursively', () => {
    const sanitized = sanitizeTemplates(source)

    expect(JSON.stringify(sanitized)).not.toContain('excel_path')
    expect(JSON.stringify(sanitized)).not.toContain('output')
    expect(JSON.stringify(sanitized)).not.toMatch(/\.xlsx?|\.png/)
    expect(sanitized.templates[0].series?.[0].x_col_index).toBe(0)
    expect(() => assertPublicTemplates(sanitized)).not.toThrow()
  })
})
