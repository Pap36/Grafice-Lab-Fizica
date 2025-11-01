# Grafice-Lab-Fizica

Utility script for fitting linear models to physics lab data stored in Excel spreadsheets. The script reads a workbook, applies cosmetic scaling factors to axis labels, performs a linear regression, and exports a PNG plot alongside the source spreadsheet.

## Installation

### macOS
- Install Homebrew if it is not already available:
	```bash
	/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
	```
- Install Python 3 and ensure `pip` is available:
	```bash
	brew install python@3.11
	python3 -m pip install --upgrade pip
	```
- (Recommended) Create and activate a virtual environment in the project folder:
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
- Install the project dependencies:
	```bash
	pip install -r requirements.txt
	```

### Windows
- Download and install the latest Python 3 release from [python.org](https://www.python.org/downloads/) and ensure the “Add python.exe to PATH” option is selected. Pip ships with the installer; no separate step is required.
- Open PowerShell in the project directory and (optionally) create a virtual environment:
	```powershell
	python -m venv .venv
	.\.venv\Scripts\Activate
	```
- Install the dependencies from `requirements.txt`:
	```powershell
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt
	```

## How to Use
- Place your Excel workbooks (`.xls`/`.xlsx`) in the repository or reference them via the template `excel_path` field.
- Run the script from the project directory (activate the virtual environment first if you created one):
	```bash
	python main.py
	```
- Choose whether to apply a saved template. Templates can define column indices, labels, scaling exponents, and a dedicated Excel file path. When the template’s file path exists, the PNG plot is stored in the same directory as the spreadsheet.
- If no template is used or the template path is missing, pick an Excel file from the presented list, follow the prompts for column selection and cosmetic options, then inspect the generated plot and statistics in the terminal.
- The script saves a high-resolution plot named `<excel_stem>_fit.png`; it appears next to the source Excel file and also opens in a Matplotlib window for immediate review.

## Add a Template (example + explanation)

Templates live in `fit_templates.json` under the top-level `templates` array. Each entry controls what columns to use, labels/units, cosmetic exponents, and whether to show slope/intercept. You can also point directly to an Excel file so you pick the template and go.

Example entry:

```json
{
	"name": "Axe paralele – Moment de inertie",
	"excel_path": "Axe paralele/axe_paralele.xlsx",
	"x_col_index": 5,
	"y_col_index": 4,
	"x_label": "a^2",
	"y_label": "\\frac{CT^2}{4 \\pi^2}",
	"x_unit": "m^2",
	"y_unit": "N \\cdot m \\cdot s^2",
	"x_exponent": -3,
	"y_exponent": -3,
	"show_slope": true,
	"slope_label": "m",
	"slope_precision": 5,
	"slope_unit": "kg",
	"slope_exponent": -3,
	"show_intercept": true,
	"intercept_label": "I_z",
	"intercept_precision": 5,
	"intercept_exponent": -3,
	"intercept_unit": "kg \\cdot m^2",
	"stats_pos": "bottom-right"
}
```

Field-by-field explanation:

- `name`: Display name shown in the template picker.
- `excel_path`: Optional path to the workbook for this template. Can be relative to where you run `python main.py` (e.g., `Axe paralele/axe_paralele.xlsx`) or an absolute path. If present and the file exists, the script opens this file automatically and saves the PNG next to it.
- `x_col_index`, `y_col_index`: Zero-based column indices in the sheet (0 is the first column). These are applied when using the template.
- `x_label`, `y_label`: Axis labels. Supports plain text and LaTeX math. You can use raw LaTeX (e.g., `\\frac{...}{...}`) or wrap with `$...$`. Plain text is auto-wrapped to look good in math mode when appropriate.
- `x_unit`, `y_unit`: Units displayed at the end of the axis label. In math mode they are typeset using `\\mathrm{(...)}`.
- `x_exponent`, `y_exponent`: Cosmetic exponents for tick formatting. A value of `k` displays ticks divided by `10^k` and appends `×10^k` in the axis label.
- `show_slope`, `show_intercept`: Toggle showing the fitted line’s slope and/or intercept in the on-plot stats box.
- `slope_label`, `intercept_label`: Symbols used in the stats, e.g., `m` or `I_z`. LaTeX supported.
- `slope_precision`, `intercept_precision`: Decimal places for displaying numeric values.
- `slope_unit`, `intercept_unit`: Units for the slope/intercept display, typeset with `\\mathrm{}` in math mode.
- `slope_exponent`, `intercept_exponent`: Cosmetic exponents applied to the displayed values (value is divided by `10^k`, and `·10^k` is appended).
- `stats_pos`: Position of the stats box. One of `top-right`, `top-left`, `bottom-right`, `bottom-left`.

Tips:

- Keep paths portable by using relative paths from the repository root. For user-specific absolute paths, prefer forward slashes on macOS and Windows (Python will normalize them).
- Column indices are validated at runtime; if they’re out of range for the selected sheet, you’ll be prompted to adjust.
- LaTeX in labels works with Matplotlib’s mathtext (no TeX installation required). Use double backslashes in JSON strings.

