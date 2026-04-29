import type { PlotTemplate, TemplatesFile } from '../types/templates'

const privateKeys = new Set([
  'excel_path',
  'output',
  'source_file',
  'sourceFile',
  'file_path',
  'filePath'
])

const privateArtifactPattern = /\.(xlsx?|png|jpe?g|pdf|tex|aux|log|txt)(?:$|[?#])/i

export function containsPrivateArtifactReference(value: unknown): boolean {
  return typeof value === 'string' && privateArtifactPattern.test(value.trim())
}

export function sanitizeTemplateValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value
      .map((item) => sanitizeTemplateValue(item))
      .filter((item) => item !== undefined)
  }

  if (value && typeof value === 'object') {
    const sanitized: Record<string, unknown> = {}
    for (const [key, item] of Object.entries(value)) {
      if (privateKeys.has(key)) {
        continue
      }
      if (containsPrivateArtifactReference(item)) {
        continue
      }
      const nextValue = sanitizeTemplateValue(item)
      if (nextValue !== undefined) {
        sanitized[key] = nextValue
      }
    }
    return sanitized
  }

  if (containsPrivateArtifactReference(value)) {
    return undefined
  }

  return value
}

export function sanitizeTemplates(input: TemplatesFile): TemplatesFile {
  const templates = (input.templates ?? [])
    .map((template) => sanitizeTemplateValue(template) as PlotTemplate)
    .filter((template) => template && typeof template.name === 'string')

  return { templates }
}

export function assertPublicTemplates(input: TemplatesFile): void {
  const encoded = JSON.stringify(input)
  if (privateArtifactPattern.test(encoded)) {
    throw new Error('Sanitized templates still contain private workbook, plot, or report references.')
  }
  for (const key of privateKeys) {
    if (encoded.includes(`"${key}"`)) {
      throw new Error(`Sanitized templates still contain private key: ${key}`)
    }
  }
}
