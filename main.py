#!/usr/bin/env python3
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from typing import Optional


# ---------- prompt helpers ----------
def ask_bool(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    s = input(f"{prompt} [{hint}]: ").strip().lower()
    if s == "":
        return default
    return s in {"y", "yes", "da"}

def ask_int(prompt: str, default: Optional[int] = None) -> int:
    if default is None:
        s = input(f"{prompt}: ").strip()
    else:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
    return int(s)

def ask_str(prompt: str, default: Optional[str] = None) -> str:
    if default is None:
        return input(f"{prompt}: ").strip()
    s = input(f"{prompt} [{default}]: ").strip()
    return s if s != "" else default


# ---------- LaTeX helpers ----------
def _looks_like_math(s: str) -> bool:
    s = (s or "").strip()
    return (s.startswith("\\")
            or any(ch in s for ch in "\\_^{}")
            or (s.startswith("$") and s.endswith("$")))

def axis_label_with_unit(label: str, unit: str, exponent: int = 1) -> str:
    """
    Build an axis label with the form:
        label ×10^{exp} (unit)
    The unit is always last.
    Works consistently with LaTeX and plain text.
    """
    label = (label or "").strip()
    unit  = (unit  or "").strip()
    exp_str_math = f"\\,\\cdot\\,10^{{{exponent}}}" if exponent != 0 else ""
    exp_str_plain = f" ×10^{exponent}" if exponent != 0 else ""

    # Detect LaTeX math
    if label.startswith("$") and label.endswith("$"):
        core = label[1:-1]; is_math = True
    else:
        core = label; is_math = _looks_like_math(label)

    if is_math:
        # math label
        if exponent != 0 and unit:
            return f"${core}{exp_str_math}\\,\\mathrm{{({unit})}}$"
        elif exponent != 0:
            return f"${core}{exp_str_math}$"
        elif unit:
            return f"${core}\\,\\mathrm{{({unit})}}$"
        else:
            return f"${core}$"
    else:
        # plain label
        if exponent != 0 and unit:
            return f"{label}{exp_str_plain} ({unit})"
        elif exponent != 0:
            return f"{label}{exp_str_plain}"
        elif unit:
            return f"{label} ({unit})"
        else:
            return label

def label_in_math_context(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("$") and s.endswith("$"):
        return s[1:-1]
    if _looks_like_math(s):
        return s
    return r"\mathrm{" + s.replace(" ", r"\ ") + "}"

def format_plot_title(title: Optional[str]) -> str:
    title = (title or "").strip()
    if not title:
        return ""
    if title.startswith("$") and title.endswith("$"):
        return title
    if "$" in title:
        return title
    if _looks_like_math(title):
        return f"${title}$"
    return title


def main():
    print("=== Linear Fit from Excel (cosmetic exponents, unit last) ===\n")

    # ---- Load templates (allow template to provide an excel_path) ----
    templates_path = Path("fit_templates.json")
    tmpl = {}
    used_template = False
    tmpls = []
    if templates_path.exists():
        try:
            data = json.loads(templates_path.read_text(encoding="utf-8"))
            tmpls = data.get("templates", [])
        except Exception as e:
            print(f"Warning: failed to read templates: {e}")
            tmpls = []

    if tmpls:
        print("\nTemplates available:")
        for i, t in enumerate(tmpls):
            ep = t.get("excel_path")
            extra = f" -> {ep}" if ep else ""
            print(f"  [{i}] {t.get('name','(unnamed)')}{extra}")
        use_tmpl = ask_bool("Use a template?", default=False)
        if use_tmpl:
            ti = ask_int("Select template index", 0)
            if 0 <= ti < len(tmpls):
                tmpl = tmpls[ti]
                used_template = True
                print(f"Using template: {tmpl.get('name','(unnamed)')}")
            else:
                print("Template index out of range. Continuing without.")

    # ---- Select Excel file(s) (template may have provided a path or per-series files) ----
    path = None
    # detect multi-file template usage: any series or fit declares its own excel_path, or template provides multi_files
    multi_file_mode = False
    if used_template:
        if tmpl.get("multi_files"):
            multi_file_mode = True
        else:
            series_cfg_tmp = tmpl.get("series")
            fits_cfg_tmp = tmpl.get("fits")
            if isinstance(series_cfg_tmp, list):
                for s in series_cfg_tmp:
                    if s.get("excel_path"):
                        multi_file_mode = True
                        break
            if not multi_file_mode and isinstance(fits_cfg_tmp, list):
                for f in fits_cfg_tmp:
                    if f.get("excel_path"):
                        multi_file_mode = True
                        break

    if not multi_file_mode:
        if used_template and tmpl.get("excel_path"):
            candidate = Path(tmpl.get("excel_path"))
            if candidate.exists():
                path = candidate
                print(f"\nUsing Excel file from template: {path}")
            else:
                print(f"\nWarning: template specifies excel_path '{candidate}', but that file was not found.\nFalling back to manual file selection.")

        if path is None:
            excel_files = sorted(Path(".").glob("*.xls*"))
            if not excel_files:
                print("No Excel files (.xls/.xlsx) found here."); sys.exit(1)

            print("Available Excel files:")
            for i, f in enumerate(excel_files):
                print(f"  [{i}] {f.name}")
            file_idx = ask_int("\nSelect file index", 0)
            if not (0 <= file_idx < len(excel_files)):
                print("ERROR: index out of range."); sys.exit(1)
            path = excel_files[file_idx]
            print(f"\nReading: {path.name}")

        # ---- Read single Excel ----
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"ERROR reading Excel: {e}"); sys.exit(1)
        if df.empty:
            print("ERROR: sheet is empty."); sys.exit(1)
    else:
        # multi-file mode: we'll load per-series / per-fit files as needed
        print("Template requests multiple Excel files; files will be read per-series or per-fit.")

    # ---- Columns / Series selection ----
    series_cfg = tmpl.get("series") if used_template else None
    multi_series = isinstance(series_cfg, list) and len(series_cfg) > 0
    legend_label_override = tmpl.get("plot_label") if used_template else None

    # If we are in single-file mode, show columns and let existing flow work
    if not multi_file_mode:
        print("\nColumns (0-based):")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")

        if not multi_series:
            # Single-series path (backward compatible)
            if used_template:
                x_idx = int(tmpl.get("x_col_index"))
                y_idx = int(tmpl.get("y_col_index"))
                print(f"Using template column indices: X={x_idx}, Y={y_idx}")
            else:
                x_idx = ask_int("Index for X column", tmpl.get("x_col_index"))
                y_idx = ask_int("Index for Y column", tmpl.get("y_col_index"))

            if not (0 <= x_idx < len(df.columns) and 0 <= y_idx < len(df.columns)):
                print("ERROR: column index out of range."); sys.exit(1)

            x = pd.to_numeric(df.iloc[:, x_idx], errors="coerce").to_numpy()
            y = pd.to_numeric(df.iloc[:, y_idx], errors="coerce").to_numpy()
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 2:
                print("ERROR: need at least two valid numeric pairs."); sys.exit(1)
            series_list = [
                {
                    "label": tmpl.get("series_label") or tmpl.get("name") or "Series 1",
                    "x": x,
                    "y": y,
                    "x_idx": x_idx,
                    "y_idx": y_idx,
                    "color": None,
                    "linestyle": "-",
                    "marker": "o",
                    "linewidth": 2.0,
                }
            ]
        else:
            # Multi-series from template (single-file)
            print("\nColumns (0-based):")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")
            series_list = []
            for k, s in enumerate(series_cfg, start=1):
                try:
                    sx = int(s.get("x_col_index"))
                    sy = int(s.get("y_col_index"))
                except Exception:
                    print("ERROR: series definitions must include integer x_col_index and y_col_index.")
                    sys.exit(1)
                if not (0 <= sx < len(df.columns) and 0 <= sy < len(df.columns)):
                    print(f"ERROR: series[{k}] column index out of range.")
                    sys.exit(1)
                row_start = s.get("row_start")
                row_end = s.get("row_end")
                sel = slice(row_start, row_end) if (row_start is not None or row_end is not None) else slice(None)
                xv = pd.to_numeric(df.iloc[sel, sx], errors="coerce").to_numpy()
                yv = pd.to_numeric(df.iloc[sel, sy], errors="coerce").to_numpy()
                mask = np.isfinite(xv) & np.isfinite(yv)
                xv, yv = xv[mask], yv[mask]
                if len(xv) < 2:
                    print(f"ERROR: series[{k}] needs at least two valid numeric pairs.")
                    sys.exit(1)
                series_list.append({
                    "label": s.get("label") or f"Series {k}",
                    "x": xv,
                    "y": yv,
                    "x_idx": sx,
                    "y_idx": sy,
                    "color": s.get("color"),
                    "linestyle": s.get("linestyle", "-"),
                    "marker": s.get("marker", "o"),
                    "linewidth": float(s.get("linewidth", 2.0)),
                })
    else:
        # Multi-file mode: read each series from its own excel file (or template-provided file path)
        if not multi_series:
            print("ERROR: template multi-file mode requires a 'series' list with per-series excel_path entries.")
            sys.exit(1)
        series_list = []
        for k, s in enumerate(series_cfg, start=1):
            excel_p = Path(s.get("excel_path") or tmpl.get("excel_path") or "")
            if not excel_p or not excel_p.exists():
                print(f"ERROR: series[{k}] specifies missing excel_path: {excel_p}"); sys.exit(1)
            try:
                df_s = pd.read_excel(excel_p)
            except Exception as e:
                print(f"ERROR reading Excel for series[{k}]: {e}"); sys.exit(1)
            try:
                sx = int(s.get("x_col_index"))
                sy = int(s.get("y_col_index"))
            except Exception:
                print("ERROR: series definitions must include integer x_col_index and y_col_index.")
                sys.exit(1)
            if not (0 <= sx < len(df_s.columns) and 0 <= sy < len(df_s.columns)):
                print(f"ERROR: series[{k}] column index out of range for file {excel_p}.")
                sys.exit(1)
            row_start = s.get("row_start")
            row_end = s.get("row_end")
            sel = slice(row_start, row_end) if (row_start is not None or row_end is not None) else slice(None)
            xv = pd.to_numeric(df_s.iloc[sel, sx], errors="coerce").to_numpy()
            yv = pd.to_numeric(df_s.iloc[sel, sy], errors="coerce").to_numpy()
            mask = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[mask], yv[mask]
            if len(xv) < 2:
                print(f"ERROR: series[{k}] in file {excel_p} needs at least two valid numeric pairs.")
                sys.exit(1)
            series_list.append({
                "label": s.get("label") or f"Series {k}",
                "x": xv,
                "y": yv,
                "x_idx": sx,
                "y_idx": sy,
                "color": s.get("color"),
                "linestyle": s.get("linestyle", "-"),
                "marker": s.get("marker", "o"),
                "linewidth": float(s.get("linewidth", 2.0)),
                "source_file": str(excel_p)
            })

    # (Fit is computed later per-series only when plot_mode == 'fit' or when explicit fits are provided)
    if legend_label_override and len(series_list) == 1:
        series_list[0]["label"] = legend_label_override

    # ---- Optional explicit fits (can co-exist with series) ----
    fits_cfg = tmpl.get("fits") if used_template else None
    fits_list = []

    def extract_xy_from_df(gdf, col_x: int, col_y: int, row_start=None, row_end=None):
        sel = slice(row_start, row_end) if (row_start is not None or row_end is not None) else slice(None)
        xv = pd.to_numeric(gdf.iloc[sel, col_x], errors="coerce").to_numpy()
        yv = pd.to_numeric(gdf.iloc[sel, col_y], errors="coerce").to_numpy()
        mask = np.isfinite(xv) & np.isfinite(yv)
        return xv[mask], yv[mask]

    def linear_fit(xv, yv, *, force_origin: bool = False):
        if force_origin:
            denom = np.sum(xv * xv)
            if denom == 0:
                print("ERROR: cannot enforce origin fit when all X values are zero."); sys.exit(1)
            slope = float(np.sum(xv * yv) / denom)
            return slope, 0.0
        return np.polyfit(xv, yv, 1)

    if isinstance(fits_cfg, list):
        for k, f in enumerate(fits_cfg, start=1):
            try:
                fx = int(f.get("x_col_index"))
                fy = int(f.get("y_col_index"))
            except Exception:
                print("ERROR: fits definitions must include integer x_col_index and y_col_index.")
                sys.exit(1)
            # for multi-file fits, allow per-fit excel_path
            if not multi_file_mode:
                if not (0 <= fx < len(df.columns) and 0 <= fy < len(df.columns)):
                    print(f"ERROR: fits[{k}] column index out of range.")
                    sys.exit(1)
                xv, yv = extract_xy_from_df(df, fx, fy, row_start=f.get("row_start"), row_end=f.get("row_end"))
            else:
                excel_p = Path(f.get("excel_path") or tmpl.get("excel_path") or "")
                if not excel_p or not excel_p.exists():
                    print(f"ERROR: fits[{k}] specifies missing excel_path: {excel_p}"); sys.exit(1)
                try:
                    df_f = pd.read_excel(excel_p)
                except Exception as e:
                    print(f"ERROR reading Excel for fits[{k}]: {e}"); sys.exit(1)
                if not (0 <= fx < len(df_f.columns) and 0 <= fy < len(df_f.columns)):
                    print(f"ERROR: fits[{k}] column index out of range for file {excel_p}.")
                    sys.exit(1)
                xv, yv = extract_xy_from_df(df_f, fx, fy, row_start=f.get("row_start"), row_end=f.get("row_end"))
            if len(xv) < 2:
                print(f"ERROR: fits[{k}] needs at least two valid numeric pairs.")
                sys.exit(1)
            fits_list.append({
                "label": f.get("label") or f"Fit {k}",
                "x": xv,
                "y": yv,
                "x_idx": fx,
                "y_idx": fy,
                "color": f.get("color"),
                "linestyle": f.get("linestyle", "--"),
                "linewidth": float(f.get("linewidth", 2.0)),
                "scatter": bool(f.get("scatter", False)),
            })

    # ---- Labels, units, options ----
    # For multi-series, use common labels from template or fallback to first series' column names
    # Determine axis labels: prefer explicit template labels, else fall back to first series label
    if not multi_series:
        if not multi_file_mode:
            x_label_in = tmpl.get("x_label", str(df.columns[series_list[0]["x_idx"]]))
            y_label_in = tmpl.get("y_label", str(df.columns[series_list[0]["y_idx"]]))
        else:
            x_label_in = tmpl.get("x_label", series_list[0].get("label") or "x")
            y_label_in = tmpl.get("y_label", series_list[0].get("label") or "y")
    else:
        # multi-series: use template-provided labels or fall back to first series label
        x_label_in = tmpl.get("x_label", series_list[0].get("label") or "x")
        y_label_in = tmpl.get("y_label", series_list[0].get("label") or "y")
    x_unit = tmpl.get("x_unit", "")
    y_unit = tmpl.get("y_unit", "")
    x_exp = int(tmpl.get("x_exponent", 1))
    y_exp = int(tmpl.get("y_exponent", 1))
    slope_label = tmpl.get("slope_label", "m")
    slope_unit = tmpl.get("slope_unit", "")
    slope_exp = int(tmpl.get("slope_exponent", 1))
    slope_prec = int(tmpl.get("slope_precision", 5))
    intercept_label = tmpl.get("intercept_label", "b")
    intercept_unit = tmpl.get("intercept_unit", "")
    intercept_exp = int(tmpl.get("intercept_exponent", 1))
    intercept_prec = int(tmpl.get("intercept_precision", 5))
    show_slope = bool(tmpl.get("show_slope", True))
    show_intercept = bool(tmpl.get("show_intercept", True))
    pos = (tmpl.get("stats_pos") or "bottom-right").lower()
    force_origin = bool(tmpl.get("force_through_origin"))
    plot_title_cfg = tmpl.get("plot_name") if used_template else None
    # Axis lower-bound controls
    x_allow_negative = bool(tmpl.get("x_allow_negative", True))
    y_allow_negative = bool(tmpl.get("y_allow_negative", True))
    x_start_at_zero = bool(tmpl.get("x_start_at_zero", False))
    y_start_at_zero = bool(tmpl.get("y_start_at_zero", False))

    # Optional axis upper-bound controls
    y_max = None
    if "y_max" in tmpl and tmpl.get("y_max") is not None:
        try:
            y_max = float(tmpl.get("y_max"))
        except Exception:
            print("Warning: template y_max is not a valid number; ignoring.")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_mode = (tmpl.get("plot_mode") or "fit").strip().lower()
    last_fit = None  # store last (m, b, x, y) for single-series annotation
    if plot_mode not in {"fit", "lines"}:
        print("Warning: unknown plot_mode; defaulting to 'fit'.")
        plot_mode = "fit"

    # Determine global x-range to extend fit lines
    all_x_values = []
    for s in series_list:
        all_x_values.append(s["x"])
    for f in fits_list:
        all_x_values.append(f["x"])
    global_xmin = min((np.min(a) for a in all_x_values), default=None)
    global_xmax = max((np.max(a) for a in all_x_values), default=None)
    # If axis requested to start at zero, extend global range to include 0 for fit line drawing
    if x_start_at_zero and global_xmin is not None and global_xmin > 0:
        global_xmin = 0.0

    # If there are explicit fits, always draw series as lines/markers
    if fits_list:
        for s in series_list:
            base_label = s["label"]
            if used_template and tmpl.get("hide_base_series_legend", False):
                base_label = None
            if used_template and tmpl.get("series_scatter_only", False):
                ax.scatter(s["x"], s["y"], s=30, color=s["color"], marker=s["marker"], label=base_label)
            else:
                ax.plot(s["x"], s["y"], linestyle=s["linestyle"], marker=s["marker"],
                        linewidth=s["linewidth"], color=s["color"], label=base_label)
    else:
        # Back-compat behavior when explicit fits not provided
        for s in series_list:
            xv, yv = s["x"], s["y"]
            label = s["label"]
            color = s["color"]
            ls = s["linestyle"]
            marker = s["marker"]
            lw = s["linewidth"]
            if plot_mode == "fit":
                m, b = linear_fit(xv, yv, force_origin=force_origin)
                x_line = np.linspace(np.min(xv), np.max(xv), 200)
                y_line = m * x_line + b
                ax.scatter(xv, yv, s=30, color=color, label=None)
                ax.plot(x_line, y_line, linewidth=lw, linestyle=ls, color=color, label=label)
                last_fit = (m, b, xv, yv, label)
            else:
                ax.plot(xv, yv, linestyle=ls, marker=marker, linewidth=lw, color=color, label=label)

    # Draw explicit fits and extend lines across global x-range
    for f in fits_list:
        xv, yv = f["x"], f["y"]
        label = f["label"]
        color = f["color"]
        ls = f["linestyle"]
        lw = f["linewidth"]
        m, b = linear_fit(xv, yv, force_origin=force_origin)
        if global_xmin is not None and global_xmax is not None and global_xmax > global_xmin:
            x_line = np.linspace(global_xmin, global_xmax, 400)
        else:
            x_line = np.linspace(np.min(xv), np.max(xv), 200)
        y_line = m * x_line + b
        if f.get("scatter"):
            ax.scatter(xv, yv, s=30, color=color, label=None)
        ax.plot(x_line, y_line, linewidth=lw, linestyle=ls, color=color, label=label)
        last_fit = (m, b, xv, yv, label)

    def make_div_formatter(divisor: float):
        if divisor == 0 or divisor == 1:
            return FuncFormatter(lambda val, _pos: f"{val:g}")
        inv = 1.0 / divisor
        return FuncFormatter(lambda val, _pos: f"{val*inv:g}")

    ax.xaxis.set_major_formatter(make_div_formatter(10 ** x_exp))
    ax.yaxis.set_major_formatter(make_div_formatter(10 ** y_exp))

    ax.set_xlabel(axis_label_with_unit(x_label_in, x_unit, x_exp))
    ax.set_ylabel(axis_label_with_unit(y_label_in, y_unit, y_exp))
    ax.grid(True, linestyle="--", alpha=0.4)
    title_text = format_plot_title(plot_title_cfg)
    if title_text:
        ax.set_title(title_text)

    # ---- Annotation / Legend ----
    lines = []
    if plot_mode == "fit" and len(series_list) == 1 and last_fit is not None:
        m, b, xv, yv, label = last_fit
        if show_slope:
            slope_lbl = label_in_math_context(slope_label)
            if slope_exp != 0:
                display_val = m / (10 ** slope_exp)
                exp_str = f"\\cdot 10^{{{slope_exp}}}"
            else:
                display_val = m
                exp_str = ""
            unit_str = f"\\;\\mathrm{{({slope_unit})}}" if slope_unit else ""
            lines.append(f"${slope_lbl} = {display_val:.{slope_prec}f}{exp_str}{unit_str}$")

        if show_intercept:
            intercept_lbl = label_in_math_context(intercept_label)
            if intercept_exp != 0:
                display_val = b / (10 ** intercept_exp)
                exp_str = f"\\cdot 10^{{{intercept_exp}}}"
            else:
                display_val = b
                exp_str = ""
            unit_str = f"\\;\\mathrm{{({intercept_unit})}}" if intercept_unit else ""
            lines.append(f"${intercept_lbl} = {display_val:.{intercept_prec}f}{exp_str}{unit_str}$")

    pos_map = {
        "top-right": (0.98, 0.95, "right", "top"),
        "top-left": (0.02, 0.95, "left", "top"),
        "bottom-right": (0.98, 0.05, "right", "bottom"),
        "bottom-left": (0.02, 0.05, "left", "bottom")
    }
    x_anchor, y_anchor, ha, va = pos_map.get(pos, (0.98, 0.95, "right", "top"))

    if lines:
        ax.text(
            x_anchor, y_anchor,
            "\n".join(lines),
            transform=ax.transAxes,
            va=va, ha=ha,
            fontsize=11, color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="black",
                      linewidth=0.8, alpha=0.95)
        )

    # Apply axis lower bounds if requested
    if (not x_allow_negative) or x_start_at_zero:
        try:
            left, right = ax.get_xlim()
            ax.set_xlim(left=0, right=right)
        except Exception:
            ax.set_xlim(left=0)
    if (not y_allow_negative) or y_start_at_zero:
        try:
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom=0, top=top)
        except Exception:
            ax.set_ylim(bottom=0)

    # Apply optional y-axis max if requested
    if y_max is not None:
        try:
            bottom, _top = ax.get_ylim()
            if y_max > bottom:
                ax.set_ylim(bottom=bottom, top=y_max)
        except Exception:
            pass

    # legend when multiple labeled items or any explicit labels
    if True:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    plt.tight_layout()
    if path is None:
        # derive output name from template or from series source filenames
        out_name = tmpl.get("output")
        if not out_name:
            parts = []
            for s in (tmpl.get("series") or []):
                p = s.get("excel_path") or ""
                if p:
                    parts.append(Path(p).stem)
            stem = "_".join(parts) if parts else "combined"
            out_name = f"{stem}_fit.png"
        out_path = Path(out_name)
    else:
        out_path = path.with_name(f"{path.stem}_fit.png")
    plt.savefig(out_path, dpi=150)
    if plot_mode == "fit" and len(series_list) == 1 and last_fit is not None and not fits_list:
        m, b, xv, yv, label = last_fit
        y_fit = m * xv + b
        ss_res = np.sum((yv - y_fit) ** 2)
        ss_tot = np.sum((yv - np.mean(yv)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"\nFit: y = {m:.6f} * x + {b:.6f}  (R^2 = {r2:.6f})")
    elif fits_list:
        print("\nFits (one per defined fit):")
        for f in fits_list:
            xv, yv = f["x"], f["y"]
            m, b = linear_fit(xv, yv, force_origin=force_origin)
            y_fit = m * xv + b
            ss_res = np.sum((yv - y_fit) ** 2)
            ss_tot = np.sum((yv - np.mean(yv)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"  - {f['label']}: y = {m:.6f} * x + {b:.6f}  (R^2 = {r2:.6f})")
    print(f"Plot saved to: {out_path.resolve()}\n")
    plt.show()


if __name__ == "__main__":
    main()
