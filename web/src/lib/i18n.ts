export type AppLanguage = 'en' | 'ro'

export interface AppTranslations {
  code: AppLanguage
  languageLabel: string
  english: string
  romanian: string
  uploadFile: string
  resetPlot: string
  source: string
  file: string
  noFileSelected: string
  sheet: string
  templateJson: string
  templateCsv: string
  import: string
  plots: string
  data: string
  formula: string
  label: string
  draw: string
  column: string
  x: string
  y: string
  startRow: string
  endRow: string
  fit: string
  forceOrigin: string
  startAtZero: string
  extrapolate: string
  points: string
  stats: string
  show: string
  baseline: string
  amplitude: string
  mean: string
  std: string
  coefficients: string
  coefficient: string
  rate: string
  slope: string
  intercept: string
  degree: string
  step: string
  color: string
  line: string
  marker: string
  expression: string
  latex: string
  min: string
  max: string
  width: string
  legendInfo: string
  info: string
  legendInfoEmpty: string
  fileData: string
  row: string
  axes: string
  title: string
  xLabel: string
  yLabel: string
  xUnit: string
  yUnit: string
  xExp: string
  yExp: string
  xTickStep: string
  yTickStep: string
  xTickPrecision: string
  yTickPrecision: string
  xStartsZero: string
  yStartsZero: string
  appTitle: string
  noUploadedFileInMemory: string
  png: string
  pngPreview: string
  pngPreviewHelp: string
  exportFrame: string
  exportFrameHelp: string
  downloadPng: string
  cancel: string
  resizePanels: string
  resizeSettingsPanel: string
  resizeParametersPanel: string
  uploadFileToEditData: string
  remove: string
  collapsePanel: string
  expandPanel: string
  settingsPanel: string
  plotterPanel: string
  parameters: string
  fitEquations: string
  curveEquations: string
  noEquationsYet: string
  legendInfoDivider: string
  interactivePlotName: string
  addPlotToBegin: string
  uploadFileAndChooseSheet: string
  sourceOption: {
    workbook: string
    formula: string
  }
  drawMode: {
    lines: string
    steps: string
    fit: string
  }
  fitModel: {
    linear: string
    polynomial: string
    exponential: string
    gaussian: string
  }
  stepOption: {
    pre: string
    post: string
    mid: string
  }
  lineStyle: {
    solid: string
    dash: string
    dot: string
    dashDot: string
  }
  markerStyle: {
    circle: string
    square: string
    triangle: string
    cross: string
    none: string
  }
  statsPositionOption: {
    topRight: string
    topLeft: string
    bottomRight: string
    bottomLeft: string
  }
  plot: (displayIndex: number) => string
  formulaItem: (displayIndex: number) => string
  infoItem: (displayIndex: number) => string
  rowsColumns: (rowCount: number, columnCount: number) => string
  downloadedTemplateSettings: (format: string) => string
  importedTemplateSettings: (fileName?: string) => string
  columnNameAria: (displayIndex: number) => string
  cellAria: (rowDisplayIndex: number, columnDisplayIndex: number) => string
}

export const languageOptions: AppLanguage[] = ['en', 'ro']

const translations: Record<AppLanguage, AppTranslations> = {
  en: {
    code: 'en',
    languageLabel: 'Language',
    english: 'English',
    romanian: 'Romana',
    uploadFile: 'Upload File',
    resetPlot: 'Reset plot',
    source: 'Source',
    file: 'File',
    noFileSelected: 'No file selected',
    sheet: 'Sheet',
    templateJson: 'Template JSON',
    templateCsv: 'Template CSV',
    import: 'Import',
    plots: 'Plots',
    data: 'Data',
    formula: 'Formula',
    label: 'Label',
    draw: 'Draw',
    column: 'Column',
    x: 'X',
    y: 'Y',
    startRow: 'Start row',
    endRow: 'End row',
    fit: 'Fit',
    forceOrigin: 'Force origin',
    startAtZero: 'Start at 0',
    extrapolate: 'Extrapolate',
    points: 'Points',
    stats: 'Stats',
    show: 'Show',
    baseline: 'Baseline',
    amplitude: 'Amplitude',
    mean: 'Mean',
    std: 'Std',
    coefficients: 'Coefficients',
    coefficient: 'Coefficient',
    rate: 'Rate',
    slope: 'Slope',
    intercept: 'Intercept',
    degree: 'Degree',
    step: 'Step',
    color: 'Color',
    line: 'Line',
    marker: 'Marker',
    expression: 'Expression',
    latex: 'LaTeX',
    min: 'Min',
    max: 'Max',
    width: 'Width',
    legendInfo: 'Legend Info',
    info: 'Info',
    legendInfoEmpty: 'Add legend-only notes. They appear under a separate legend divider and do not draw data on the plot.',
    fileData: 'File Data',
    row: 'Row',
    axes: 'Axes',
    title: 'Title',
    xLabel: 'X label',
    yLabel: 'Y label',
    xUnit: 'X unit',
    yUnit: 'Y unit',
    xExp: 'X exp',
    yExp: 'Y exp',
    xTickStep: 'X step',
    yTickStep: 'Y step',
    xTickPrecision: 'X decimals',
    yTickPrecision: 'Y decimals',
    xStartsZero: 'X starts zero',
    yStartsZero: 'Y starts zero',
    appTitle: 'Physics Lab Plotter',
    noUploadedFileInMemory: 'No uploaded file in memory',
    png: 'PNG',
    pngPreview: 'PNG Preview',
    pngPreviewHelp: 'Drag the title, legend, and stats box into place. Resize the export frame to choose the area saved to PNG.',
    exportFrame: 'Export frame',
    exportFrameHelp: 'Drag the frame or its edges to crop the exported image.',
    downloadPng: 'Download PNG',
    cancel: 'Cancel',
    resizePanels: 'Resize panels',
    resizeSettingsPanel: 'Resize settings panel',
    resizeParametersPanel: 'Resize parameters panel',
    uploadFileToEditData: 'Upload a file to edit data.',
    remove: 'Remove',
    collapsePanel: 'Collapse panel',
    expandPanel: 'Expand panel',
    settingsPanel: 'Settings',
    plotterPanel: 'Plotter',
    parameters: 'Parameters',
    fitEquations: 'Fits',
    curveEquations: 'Curves',
    noEquationsYet: 'Fit equations and formula curves will appear here.',
    legendInfoDivider: '──────── Legend Info ────────',
    interactivePlotName: 'Interactive plot',
    addPlotToBegin: 'Add a plot to begin.',
    uploadFileAndChooseSheet: 'upload a file and choose a sheet.',
    sourceOption: {
      workbook: 'File data',
      formula: 'Formula'
    },
    drawMode: {
      lines: 'lines',
      steps: 'steps',
      fit: 'fit'
    },
    fitModel: {
      linear: 'linear',
      polynomial: 'polynomial',
      exponential: 'exponential',
      gaussian: 'gaussian'
    },
    stepOption: {
      pre: 'Pre',
      post: 'Post',
      mid: 'Mid'
    },
    lineStyle: {
      solid: 'Solid',
      dash: 'Dash',
      dot: 'Dot',
      dashDot: 'Dash dot'
    },
    markerStyle: {
      circle: 'Circle',
      square: 'Square',
      triangle: 'Triangle',
      cross: 'Cross',
      none: 'None'
    },
    statsPositionOption: {
      topRight: 'Top right',
      topLeft: 'Top left',
      bottomRight: 'Bottom right',
      bottomLeft: 'Bottom left'
    },
    plot: (displayIndex) => `Plot ${displayIndex}`,
    formulaItem: (displayIndex) => `Formula ${displayIndex}`,
    infoItem: (displayIndex) => `Info ${displayIndex}`,
    rowsColumns: (rowCount, columnCount) => `${rowCount} rows, ${columnCount} columns`,
    downloadedTemplateSettings: (format) => `Downloaded ${format} template settings.`,
    importedTemplateSettings: (fileName) => `Imported template settings.${fileName ? ` Imported template was saved for ${fileName}.` : ''}`,
    columnNameAria: (displayIndex) => `Column ${displayIndex} name`,
    cellAria: (rowDisplayIndex, columnDisplayIndex) => `Row ${rowDisplayIndex}, column ${columnDisplayIndex}`
  },
  ro: {
    code: 'ro',
    languageLabel: 'Limba',
    english: 'English',
    romanian: 'Romana',
    uploadFile: 'Incarca fisier',
    resetPlot: 'Reseteaza graficul',
    source: 'Sursa',
    file: 'Fisier',
    noFileSelected: 'Niciun fisier selectat',
    sheet: 'Foaie',
    templateJson: 'Sablon JSON',
    templateCsv: 'Sablon CSV',
    import: 'Importa',
    plots: 'Grafice',
    data: 'Date',
    formula: 'Formula',
    label: 'Eticheta',
    draw: 'Afisare',
    column: 'Coloana',
    x: 'X',
    y: 'Y',
    startRow: 'Rand initial',
    endRow: 'Rand final',
    fit: 'Ajustare',
    forceOrigin: 'Forteaza originea',
    startAtZero: 'Porneste de la 0',
    extrapolate: 'Extrapoleaza',
    points: 'Puncte',
    stats: 'Statistici',
    show: 'Afiseaza',
    baseline: 'Baza',
    amplitude: 'Amplitudine',
    mean: 'Medie',
    std: 'Abatere',
    coefficients: 'Coeficienti',
    coefficient: 'Coeficient',
    rate: 'Rata',
    slope: 'Panta',
    intercept: 'Interceptie',
    degree: 'Grad',
    step: 'Treapta',
    color: 'Culoare',
    line: 'Linie',
    marker: 'Marcaj',
    expression: 'Expresie',
    latex: 'LaTeX',
    min: 'Minim',
    max: 'Maxim',
    width: 'Grosime',
    legendInfo: 'Informatii legenda',
    info: 'Info',
    legendInfoEmpty: 'Adauga note doar pentru legenda. Ele apar sub un separator dedicat si nu deseneaza date pe grafic.',
    fileData: 'Date fisier',
    row: 'Rand',
    axes: 'Axe',
    title: 'Titlu',
    xLabel: 'Eticheta X',
    yLabel: 'Eticheta Y',
    xUnit: 'Unitate X',
    yUnit: 'Unitate Y',
    xExp: 'Exponent X',
    yExp: 'Exponent Y',
    xTickStep: 'Pas X',
    yTickStep: 'Pas Y',
    xTickPrecision: 'Zecimale X',
    yTickPrecision: 'Zecimale Y',
    xStartsZero: 'X porneste din zero',
    yStartsZero: 'Y porneste din zero',
    appTitle: 'Plotter laborator de fizica',
    noUploadedFileInMemory: 'Niciun fisier incarcat in memorie',
    png: 'PNG',
    pngPreview: 'Previzualizare PNG',
    pngPreviewHelp: 'Trage titlul, legenda si caseta de statistici unde doresti. Redimensioneaza cadrul de export pentru a alege zona salvata ca PNG.',
    exportFrame: 'Cadru export',
    exportFrameHelp: 'Trage cadrul sau marginile lui pentru a decupa imaginea exportata.',
    downloadPng: 'Descarca PNG',
    cancel: 'Anuleaza',
    resizePanels: 'Redimensioneaza panourile',
    resizeSettingsPanel: 'Redimensioneaza panoul de setari',
    resizeParametersPanel: 'Redimensioneaza panoul de parametri',
    uploadFileToEditData: 'Incarca un fisier pentru a edita datele.',
    remove: 'Elimina',
    collapsePanel: 'Restrange panoul',
    expandPanel: 'Extinde panoul',
    settingsPanel: 'Setari',
    plotterPanel: 'Plotter',
    parameters: 'Parametri',
    fitEquations: 'Ajustari',
    curveEquations: 'Curbe',
    noEquationsYet: 'Ecuatiile ajustarilor si curbelor vor aparea aici.',
    legendInfoDivider: '──────── Informatii legenda ────────',
    interactivePlotName: 'Grafic interactiv',
    addPlotToBegin: 'Adauga un grafic pentru a incepe.',
    uploadFileAndChooseSheet: 'incarca un fisier si alege o foaie.',
    sourceOption: {
      workbook: 'Date din fisier',
      formula: 'Formula'
    },
    drawMode: {
      lines: 'linii',
      steps: 'trepte',
      fit: 'ajustare'
    },
    fitModel: {
      linear: 'lineara',
      polynomial: 'polinomiala',
      exponential: 'exponentiala',
      gaussian: 'gaussiana'
    },
    stepOption: {
      pre: 'Inainte',
      post: 'Dupa',
      mid: 'Mijloc'
    },
    lineStyle: {
      solid: 'Continua',
      dash: 'Intrerupta',
      dot: 'Punctata',
      dashDot: 'Linie-punct'
    },
    markerStyle: {
      circle: 'Cerc',
      square: 'Patrat',
      triangle: 'Triunghi',
      cross: 'Cruce',
      none: 'Fara'
    },
    statsPositionOption: {
      topRight: 'Sus dreapta',
      topLeft: 'Sus stanga',
      bottomRight: 'Jos dreapta',
      bottomLeft: 'Jos stanga'
    },
    plot: (displayIndex) => `Grafic ${displayIndex}`,
    formulaItem: (displayIndex) => `Formula ${displayIndex}`,
    infoItem: (displayIndex) => `Info ${displayIndex}`,
    rowsColumns: (rowCount, columnCount) => `${rowCount} randuri, ${columnCount} coloane`,
    downloadedTemplateSettings: (format) => `Setarile sablonului ${format} au fost descarcate.`,
    importedTemplateSettings: (fileName) => `Setarile sablonului au fost importate.${fileName ? ` Sablonul a fost salvat pentru ${fileName}.` : ''}`,
    columnNameAria: (displayIndex) => `Numele coloanei ${displayIndex}`,
    cellAria: (rowDisplayIndex, columnDisplayIndex) => `Randul ${rowDisplayIndex}, coloana ${columnDisplayIndex}`
  }
}

export function detectInitialLanguage(): AppLanguage {
  if (typeof navigator !== 'undefined' && navigator.language.toLowerCase().startsWith('ro')) {
    return 'ro'
  }
  return 'en'
}

export function getTranslations(language: AppLanguage): AppTranslations {
  return translations[language]
}

export function localizeRuntimeMessage(message: string, language: AppLanguage): string {
  if (language === 'en') {
    return message
  }

  const t = translations.ro

  const plotMatch = message.match(/^Plot (\d+): (.*)$/s)
  if (plotMatch) {
    return `${t.plot(Number(plotMatch[1]))}: ${localizeRuntimeMessage(plotMatch[2], language)}`
  }

  const formulaMatch = message.match(/^Formula (\d+): (.*)$/s)
  if (formulaMatch) {
    return `${t.formulaItem(Number(formulaMatch[1]))}: ${localizeRuntimeMessage(formulaMatch[2], language)}`
  }

  const polynomialDegreeMatch = message.match(/^Polynomial degree (\d+) requires at least (\d+) points\.$/)
  if (polynomialDegreeMatch) {
    return `Gradul polinomial ${polynomialDegreeMatch[1]} necesita cel putin ${polynomialDegreeMatch[2]} puncte.`
  }

  const invalidFormulaParameterMatch = message.match(/^Invalid formula parameter name: (.+)$/)
  if (invalidFormulaParameterMatch) {
    return `Nume invalid pentru parametrul formulei: ${invalidFormulaParameterMatch[1]}`
  }

  const invalidFormulaIdentifierMatch = message.match(/^Unsupported formula identifier: (.+)$/)
  if (invalidFormulaIdentifierMatch) {
    return `Identificator de formula neacceptat: ${invalidFormulaIdentifierMatch[1]}`
  }

  switch (message) {
    case 'Add a plot to begin.':
      return t.addPlotToBegin
    case 'upload a file and choose a sheet.':
      return t.uploadFileAndChooseSheet
    case 'Please upload an .xlsx file.':
      return 'Incarca un fisier .xlsx.'
    case 'The file does not contain any readable sheets.':
      return 'Fisierul nu contine foi care pot fi citite.'
    case 'Template file is empty.':
      return 'Fisierul sablon este gol.'
    case 'Template must be an object.':
      return 'Sablonul trebuie sa fie un obiect.'
    case 'Unsupported template format.':
      return 'Formatul sablonului nu este acceptat.'
    case 'Template does not contain interactive plot settings.':
      return 'Sablonul nu contine setari pentru graficul interactiv.'
    case 'Choose both X and Y columns.':
      return 'Alege atat coloana X, cat si coloana Y.'
    case 'Selected columns are outside the uploaded sheet.':
      return 'Coloanele selectate sunt in afara foii incarcate.'
    case 'Need at least two valid numeric X/Y pairs.':
      return 'Sunt necesare cel putin doua perechi X/Y numerice valide.'
    case 'Linear fit requires matching X/Y arrays with at least two points.':
      return 'Ajustarea liniara necesita siruri X/Y corespunzatoare cu cel putin doua puncte.'
    case 'Cannot force an origin fit when all X values are zero.':
      return 'Nu poti forta ajustarea prin origine cand toate valorile X sunt zero.'
    case 'Cannot fit a line when all X values are identical.':
      return 'Nu se poate ajusta o linie cand toate valorile X sunt identice.'
    case 'Could not solve the fit because the data matrix is singular.':
      return 'Ajustarea nu a putut fi rezolvata deoarece matricea datelor este singulara.'
    case 'Polynomial fit requires matching X/Y arrays with at least two points.':
      return 'Ajustarea polinomiala necesita siruri X/Y corespunzatoare cu cel putin doua puncte.'
    case 'Exponential fit requires matching X/Y arrays with at least two points.':
      return 'Ajustarea exponentiala necesita siruri X/Y corespunzatoare cu cel putin doua puncte.'
    case 'Exponential fit requires positive finite Y values.':
      return 'Ajustarea exponentiala necesita valori Y finite si pozitive.'
    case 'Gaussian fit requires matching X/Y arrays with at least three points.':
      return 'Ajustarea gaussiana necesita siruri X/Y corespunzatoare cu cel putin trei puncte.'
    case 'Gaussian fit requires at least three finite points.':
      return 'Ajustarea gaussiana necesita cel putin trei puncte finite.'
    case 'Gaussian fit requires varying Y values.':
      return 'Ajustarea gaussiana necesita valori Y diferite.'
    case 'Formula expressions may not access object properties.':
      return 'Expresiile formulei nu pot accesa proprietati de obiect.'
    case 'Formula expression contains unsupported characters.':
      return 'Expresia formulei contine caractere neacceptate.'
    case 'Formula curves need an expression.':
      return 'Curbele de formula necesita o expresie.'
    case 'Formula curves need a valid X range.':
      return 'Curbele de formula necesita un interval X valid.'
    case 'Formula expression produced a non-finite value.':
      return 'Expresia formulei a produs o valoare nefinita.'
    default:
      return message
  }
}