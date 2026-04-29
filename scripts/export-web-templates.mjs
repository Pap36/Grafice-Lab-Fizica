#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, '..');
const inputPath = resolve(repoRoot, 'fit_templates.json');
const outputPath = resolve(repoRoot, 'web/src/data/templates.json');

const privateKeys = new Set([
  'excel_path',
  'output',
  'source_file',
  'sourceFile',
  'file_path',
  'filePath'
]);
const privateArtifactPattern = /\.(xlsx?|png|jpe?g|pdf|tex|aux|log|txt)(?:$|[?#])/i;

function containsPrivateArtifactReference(value) {
  return typeof value === 'string' && privateArtifactPattern.test(value.trim());
}

function sanitizeValue(value) {
  if (Array.isArray(value)) {
    return value.map(sanitizeValue).filter((item) => item !== undefined);
  }

  if (value && typeof value === 'object') {
    const sanitized = {};
    for (const [key, item] of Object.entries(value)) {
      if (privateKeys.has(key) || containsPrivateArtifactReference(item)) {
        continue;
      }
      const nextValue = sanitizeValue(item);
      if (nextValue !== undefined) {
        sanitized[key] = nextValue;
      }
    }
    return sanitized;
  }

  if (containsPrivateArtifactReference(value)) {
    return undefined;
  }

  return value;
}

const raw = JSON.parse(await readFile(inputPath, 'utf8'));
const templates = (raw.templates ?? [])
  .map(sanitizeValue)
  .filter((template) => template && typeof template.name === 'string');
const output = { templates };
const encoded = JSON.stringify(output, null, 2) + '\n';

if (privateArtifactPattern.test(encoded)) {
  throw new Error('Refusing to write templates: private artifact reference survived sanitization.');
}
for (const key of privateKeys) {
  if (encoded.includes(`"${key}"`)) {
    throw new Error(`Refusing to write templates: private key survived sanitization: ${key}`);
  }
}

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, encoded, 'utf8');
console.log(`Exported ${templates.length} sanitized templates to ${outputPath}`);
