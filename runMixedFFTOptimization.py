"""
Main Script for Mixed-Precision FFT Optimization
Orchestrates the complete NSGA-II optimization flow with Vivado integration.
"""

import numpy as np
import os
import shutil
import zipfile
import csv
import glob
import math

import matplotlib
matplotlib.use('Agg')           # non-interactive — safe on headless servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from globalVariablesMixedFFT import *
from objectiveEvaluationFFT import MixedPrecisionFFTProblem
from optimizationUtils import (
    MyCallback,
    SmartInitialSampling,
    BlockwiseMutation,
    StagewiseCrossover,
)


# ---------------------------------------------------------------------------
# Objective decoding helpers
# ---------------------------------------------------------------------------

def _sqnr_from_perf_error(perf_error):
    """
    Invert  perf_error = WEIGHT_PERFORMANCE / (sqnr + 1)  → SQNR (dB).
    Returns inf when perf_error == 0.
    """
    raw_pe = perf_error / WEIGHT_PERFORMANCE
    if raw_pe <= 0:
        return float('inf')
    return 1.0 / raw_pe - 1.0


def _crit_delay_ns_from_norm_latency(norm_latency, fft_size):
    """
    Invert the normalisation applied in
    objectiveEvaluationFFT._compute_actual_normalized_latency():

        norm = crit_delay_ns / REFERENCE_CLOCK_PERIOD_NS
        pipeline_factor = max(1.0, num_stages / 6.0)
        return min(norm * pipeline_factor, 10.0)

    Returns crit_delay_ns estimate (float).  When norm == 10.0 (clamped) the
    returned value is a lower bound and is flagged as '>=' in display strings.
    """
    num_stages = int(math.log2(fft_size))
    pipeline_factor = max(1.0, num_stages / 6.0)
    crit_delay = (norm_latency / pipeline_factor) * REFERENCE_CLOCK_PERIOD_NS
    return crit_delay


def _decode_objectives(obj_row, fft_size):
    """
    Decode a raw 4-element objective vector into physical quantities.

    Returns dict:
        power_W, area_luts, sqnr_db, norm_latency, crit_delay_ns, meets_timing
    """
    power_w      = obj_row[0] / WEIGHT_POWER
    area_luts    = obj_row[1] / WEIGHT_AREA
    sqnr_db      = _sqnr_from_perf_error(obj_row[2])
    norm_latency = obj_row[3] / WEIGHT_LATENCY
    crit_delay   = _crit_delay_ns_from_norm_latency(norm_latency, fft_size)
    meets_timing = crit_delay <= REFERENCE_CLOCK_PERIOD_NS

    return {
        'power_W':       power_w,
        'area_luts':     area_luts,
        'sqnr_db':       sqnr_db,
        'norm_latency':  norm_latency,
        'crit_delay_ns': crit_delay,
        'meets_timing':  meets_timing,
    }


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def setup_verilog_sources():
    """Copy Verilog source files to the working directory."""
    log_message("Setting up Verilog source files")
    wrapper_src = '../verilog_sources/mixed_precision_wrappers.v'
    wrapper_dst = os.path.join(VERILOG_SOURCES_DIR, 'mixed_precision_wrappers.v')
    if os.path.exists(wrapper_src):
        shutil.copy(wrapper_src, wrapper_dst)
        log_message("Copied wrapper file")


# ---------------------------------------------------------------------------
# CSV export — final population
# ---------------------------------------------------------------------------

def export_solutions_csv(result, fft_size, results_subdir):
    """
    Write every solution from the final population (+ Pareto front) to
    all_solutions_fft{fft_size}.csv.

    Columns:
        solution_id, fft_size,
        s0_mult, s0_add, ...,
        power_W, area_LUTs, sqnr_dB,
        norm_latency, crit_delay_ns, meets_timing,
        on_pareto_front
    """
    from fft_template_generator import FFTTemplateGenerator
    num_stages = FFTTemplateGenerator(fft_size).num_stages

    gene_headers = []
    for s in range(num_stages):
        gene_headers += [f"s{s}_mult", f"s{s}_add"]

    csv_path = os.path.join(results_subdir, f"all_solutions_fft{fft_size}.csv")

    pareto_set = set()
    if result.X is not None:
        for row in result.X:
            pareto_set.add(tuple(int(v) for v in row))

    pop   = result.pop
    all_X = pop.get("X") if pop is not None else np.empty((0, len(gene_headers)))
    all_F = pop.get("F") if pop is not None else np.empty((0, OBJECTIVES))

    if result.X is not None and result.F is not None:
        combined_X = np.vstack([result.X, all_X])
        combined_F = np.vstack([result.F, all_F])
    else:
        combined_X = all_X
        combined_F = all_F

    if len(combined_X) > 0:
        _, unique_idx = np.unique(combined_X, axis=0, return_index=True)
        combined_X = combined_X[unique_idx]
        combined_F = combined_F[unique_idx]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['solution_id', 'fft_size'] + gene_headers +
            ['power_W', 'area_LUTs', 'sqnr_dB',
             'norm_latency', 'crit_delay_ns', 'meets_timing',
             'on_pareto_front']
        )
        for idx, (x_row, f_row) in enumerate(zip(combined_X, combined_F)):
            dec      = _decode_objectives(f_row, fft_size)
            sqnr_val = dec['sqnr_db']
            on_pf    = int(tuple(int(v) for v in x_row) in pareto_set)
            writer.writerow(
                [idx, fft_size] +
                [int(v) for v in x_row] +
                [f"{dec['power_W']:.6f}",
                 int(dec['area_luts']),
                 f"{sqnr_val:.4f}" if not math.isinf(sqnr_val) else "inf",
                 f"{dec['norm_latency']:.4f}",
                 f"{dec['crit_delay_ns']:.3f}",
                 int(dec['meets_timing']),
                 on_pf]
            )

    log_message(f"Solution CSV saved → {csv_path}  ({len(combined_X)} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# Solution .txt parsing → all-generations CSV
# ---------------------------------------------------------------------------

def parse_solution_txts_to_csv(fft_size, results_subdir):
    """
    Parse every gen{G}_sol{S}.txt file that belongs to this FFT size and
    write all_generations_fft{N}.csv.

    Columns:
        generation, solution_id, fft_size,
        s0_mult, s0_add, ...,
        power_W, area_LUTs, sqnr_dB,
        norm_latency, crit_delay_ns, meets_timing,
        fp4_mult, fp8_mult, fp4_add, fp8_add
    """
    import ast as _ast, re

    num_stages   = int(math.log2(fft_size))
    gene_headers = []
    for s in range(num_stages):
        gene_headers += [f"s{s}_mult", f"s{s}_add"]

    csv_path  = os.path.join(results_subdir, f"all_generations_fft{fft_size}.csv")
    pattern   = os.path.join(RESULTS_DIR, "gen*_sol*.txt")
    txt_files = sorted(glob.glob(pattern))

    rows = []
    for fpath in txt_files:
        try:
            with open(fpath) as f:
                content = f.read()

            def _field(label):
                m = re.search(rf'^{label}\s*:\s*(.+)$', content, re.MULTILINE)
                return m.group(1).strip() if m else None

            if _field('FFT Size') is None or int(_field('FFT Size')) != fft_size:
                continue

            generation  = int(_field('Generation') or -1)
            solution_id = int(_field('Solution ID') or -1)
            chrom_raw   = _field('Chromosome')
            chromosome  = _ast.literal_eval(chrom_raw) if chrom_raw else []

            power_m  = re.search(r'Power\s*:\s*([\d.]+)\s*W',    content)
            area_m   = re.search(r'Area\s*:\s*([\d]+)\s*LUTs',   content)
            sqnr_m   = re.search(r'SQNR\s*:\s*([\d.\-]+)\s*dB',  content)
            # Vivado-derived latency fields written by _save_solution_result
            cpd_m    = re.search(r'Crit Path Delay\s*:\s*([\d.]+)\s*ns', content)
            nlat_m   = re.search(r'Norm Latency\s*:\s*([\d.]+)',  content)
            fp4mt_m  = re.search(r'FP4 Multipliers:\s*\d+\s*\(([\d.]+)%\)', content)
            fp8mt_m  = re.search(r'FP8 Multipliers:\s*\d+\s*\(([\d.]+)%\)', content)
            fp4ad_m  = re.search(r'FP4 Adders\s*:\s*\d+\s*\(([\d.]+)%\)',   content)
            fp8ad_m  = re.search(r'FP8 Adders\s*:\s*\d+\s*\(([\d.]+)%\)',   content)

            power        = float(power_m.group(1))  if power_m  else float('nan')
            area         = int(area_m.group(1))     if area_m   else -1
            sqnr         = float(sqnr_m.group(1))   if sqnr_m   else float('nan')
            crit_delay   = float(cpd_m.group(1))    if cpd_m    else float('nan')
            norm_latency = float(nlat_m.group(1))   if nlat_m   else float('nan')
            meets_timing = int(crit_delay <= REFERENCE_CLOCK_PERIOD_NS) \
                           if not math.isnan(crit_delay) else -1

            rows.append({
                'generation':   generation,
                'solution_id':  solution_id,
                'fft_size':     fft_size,
                'chromosome':   chromosome,
                'power_W':      power,
                'area_LUTs':    area,
                'sqnr_dB':      sqnr,
                'norm_latency': norm_latency,
                'crit_delay_ns': crit_delay,
                'meets_timing': meets_timing,
                'fp4_mult':     float(fp4mt_m.group(1)) if fp4mt_m else float('nan'),
                'fp8_mult':     float(fp8mt_m.group(1)) if fp8mt_m else float('nan'),
                'fp4_add':      float(fp4ad_m.group(1)) if fp4ad_m else float('nan'),
                'fp8_add':      float(fp8ad_m.group(1)) if fp8ad_m else float('nan'),
            })
        except Exception as e:
            log_message(f"  Could not parse {fpath}: {e}", level='WARN')

    rows.sort(key=lambda r: (r['generation'], r['solution_id']))

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['generation', 'solution_id', 'fft_size'] +
            gene_headers +
            ['power_W', 'area_LUTs', 'sqnr_dB',
             'norm_latency', 'crit_delay_ns', 'meets_timing',
             'fp4_mult_pct', 'fp8_mult_pct', 'fp4_add_pct', 'fp8_add_pct']
        )
        n = num_stages * 2
        for r in rows:
            chrom = (r['chromosome'] + [0] * n)[:n]
            sqnr_str = f"{r['sqnr_dB']:.4f}" if not math.isnan(r['sqnr_dB']) else "nan"
            writer.writerow(
                [r['generation'], r['solution_id'], r['fft_size']] +
                chrom +
                [f"{r['power_W']:.6f}",
                 r['area_LUTs'],
                 sqnr_str,
                 f"{r['norm_latency']:.4f}",
                 f"{r['crit_delay_ns']:.3f}",
                 r['meets_timing'],
                 f"{r['fp4_mult']:.1f}",
                 f"{r['fp8_mult']:.1f}",
                 f"{r['fp4_add']:.1f}",
                 f"{r['fp8_add']:.1f}"]
            )

    log_message(
        f"All-generations CSV saved → {csv_path}  ({len(rows)} solutions)"
    )
    return txt_files


# ---------------------------------------------------------------------------
# Compression helpers
# ---------------------------------------------------------------------------

def compress_solution_txt_files(fft_size, results_subdir, txt_files):
    if not txt_files:
        log_message("No solution .txt files to compress.", level='WARN')
        return
    zip_path = os.path.join(results_subdir, f"solution_logs_fft{fft_size}.zip")
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in txt_files:
            zf.write(fpath, os.path.basename(fpath))
    if os.path.exists(zip_path):
        deleted = 0
        for fpath in txt_files:
            try:
                os.remove(fpath)
                deleted += 1
            except OSError as e:
                log_message(f"  Warning: could not remove {fpath}: {e}", level='WARN')
        log_message(
            f"Compressed {len(txt_files)} solution log(s) → {zip_path} "
            f"({deleted} deleted)"
        )
    else:
        log_message("solution_logs zip failed — cleanup skipped.", level='WARN')


def compress_rtl_files(results_subdir, fft_size):
    zip_path    = os.path.join(results_subdir, f"rtl_fft{fft_size}.zip")
    zipped_files = []
    sim_dir      = os.path.abspath('./sim')

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:

        def _add(filepath, arcdir):
            arcname = os.path.join(arcdir, os.path.basename(filepath))
            zf.write(filepath, arcname)
            zipped_files.append(filepath)

        for f in glob.glob(os.path.join(GENERATED_DESIGNS_DIR,
                                        f"fft_{fft_size}_*.v")):
            _add(f, 'generated_designs')
        for f in glob.glob(os.path.join(results_subdir, '**', '*.v'),
                           recursive=True):
            arcname = os.path.relpath(f, results_subdir)
            zf.write(f, arcname)
            zipped_files.append(f)
        for f in glob.glob(os.path.join(sim_dir, f"tb_fft_{fft_size}_*.v")):
            _add(f, 'sim')
        for f in glob.glob(os.path.join(sim_dir,
                                        f"fft_{fft_size}_*_output.txt")):
            _add(f, 'sim')
        for f in glob.glob(os.path.join(sim_dir, f"fft_{fft_size}_*.vvp")):
            _add(f, 'sim')
        twiddle = os.path.join(sim_dir, 'twiddles_1024.txt')
        if os.path.exists(twiddle):
            zf.write(twiddle, os.path.join('sim', 'twiddles_1024.txt'))

    if os.path.exists(zip_path):
        for f in zipped_files:
            try:
                os.remove(f)
            except OSError:
                pass
        log_message(
            f"RTL zip: {len(zipped_files)} file(s) → {zip_path} "
            f"(originals deleted)"
        )
    else:
        log_message("RTL zip creation failed — cleanup skipped.", level='WARN')


# ---------------------------------------------------------------------------
# Pareto front visualisation 
# ---------------------------------------------------------------------------

# Consistent colour palette for the 4 objectives
_OBJ_COLORS = {
    'power':   '#2196F3',   # blue
    'area':    '#FF9800',   # orange
    'sqnr':    '#4CAF50',   # green
    'latency': '#E91E63',   # pink/red
}

_TIMING_OK_COLOR  = '#4CAF50'   # green  — meets clock
_TIMING_BAD_COLOR = '#E91E63'   # red    — violates clock


def _scatter_with_timing(ax, xdata, ydata, meets_timing_arr,
                         xlabel, ylabel, title, size=60):
    """
    Scatter helper that colours points green (meets timing) or red (violates).
    Also overlays a small timing-legend.
    """
    ok  = meets_timing_arr.astype(bool)
    bad = ~ok

    if ok.any():
        ax.scatter(xdata[ok],  ydata[ok],
                   c=_TIMING_OK_COLOR,  alpha=0.80, edgecolors='k',
                   linewidths=0.5, s=size, label='Meets timing', zorder=3)
    if bad.any():
        ax.scatter(xdata[bad], ydata[bad],
                   c=_TIMING_BAD_COLOR, alpha=0.80, edgecolors='k',
                   linewidths=0.5, s=size, marker='X',
                   label='Violates timing', zorder=3)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title,   fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    if ok.any() or bad.any():
        ax.legend(fontsize=8, loc='best')


def plot_pareto_front(pareto_objectives, fft_size, results_subdir, feasible=True):
    """
    Save three PNG files per FFT run:

      pareto_2d_fft{N}.png      — 6-panel grid of all pairwise 2-D projections
                                   (colour coding = timing pass/fail)
      pareto_3d_fft{N}.png      — 3-D scatter Power × Area × SQNR,
                                   colour-mapped by critical path delay
      pareto_latency_fft{N}.png — dedicated 3-panel latency dashboard
    """
    if pareto_objectives is None or len(pareto_objectives) == 0:
        log_message("No objectives to plot — Pareto plots skipped.", level='WARN')
        return

    obj = np.array(pareto_objectives)
    n   = len(obj)

    # ── Decode all 4 objectives ──────────────────────────────────────────
    power        = obj[:, 0] / WEIGHT_POWER
    area         = obj[:, 1] / WEIGHT_AREA
    perf_err     = obj[:, 2]
    norm_latency = obj[:, 3] / WEIGHT_LATENCY

    sqnr = np.array([_sqnr_from_perf_error(pe) for pe in perf_err])
    sqnr = np.where(np.isinf(sqnr), np.nan, sqnr)

    crit_delay    = np.array([
        _crit_delay_ns_from_norm_latency(nl, fft_size) for nl in norm_latency
    ])
    meets_timing  = (crit_delay <= REFERENCE_CLOCK_PERIOD_NS).astype(int)

    status_label  = "Pareto Front" if feasible else "Least-Infeasible Solutions"
    pct_ok        = 100.0 * meets_timing.sum() / n

    # ── Figure 1: 6-panel 2-D pairwise ──────────────────────────────────
    pairs = [
        (power,        area,         "Power (W)",          "Area (LUTs)",         "Power vs Area"),
        (power,        sqnr,         "Power (W)",          "SQNR (dB)",           "Power vs SQNR"),
        (power,        crit_delay,   "Power (W)",          "Crit Path Delay (ns)","Power vs Crit-Delay"),
        (area,         sqnr,         "Area (LUTs)",        "SQNR (dB)",           "Area vs SQNR"),
        (area,         crit_delay,   "Area (LUTs)",        "Crit Path Delay (ns)","Area vs Crit-Delay"),
        (sqnr,         crit_delay,   "SQNR (dB)",          "Crit Path Delay (ns)","SQNR vs Crit-Delay"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"FFT-{fft_size}  |  {status_label}  ({n} solutions)  "
        f"|  Timing pass: {pct_ok:.0f}%  "
        f"|  Clock target: {REFERENCE_CLOCK_PERIOD_NS:.1f} ns",
        fontsize=13, fontweight='bold'
    )

    for ax, (xd, yd, xl, yl, title) in zip(axes.flat, pairs):
        _scatter_with_timing(ax, xd, yd, meets_timing, xl, yl, title)

    # Add a shared reference line on every crit_delay axis (column 2)
    for ax in [axes[0, 2], axes[1, 2]]:
        ylo, yhi = ax.get_ylim()
        ax.axhline(REFERENCE_CLOCK_PERIOD_NS, color='navy', linestyle='--',
                   linewidth=1.2, label=f'Clock = {REFERENCE_CLOCK_PERIOD_NS:.1f} ns')
        ax.legend(fontsize=8, loc='best')
        ax.set_ylim(ylo, yhi)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_2d = os.path.join(results_subdir, f"pareto_2d_fft{fft_size}.png")
    fig.savefig(path_2d, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    log_message(f"2-D Pareto plot (6 panels) saved → {path_2d}")

    # ── Figure 2: 3-D scatter coloured by crit path delay ───────────────
    fig3d = plt.figure(figsize=(10, 8))
    ax3d  = fig3d.add_subplot(111, projection='3d')

    # Use crit_delay as colour; clamp colour scale to [0, 2×clock_period]
    clim_max = 2.0 * REFERENCE_CLOCK_PERIOD_NS
    c_vals   = np.clip(crit_delay, 0, clim_max)

    sc = ax3d.scatter(
        power, area, sqnr,
        c=c_vals, cmap='RdYlGn_r',          # red = slow, green = fast
        vmin=0, vmax=clim_max,
        alpha=0.85, edgecolors='k', linewidths=0.4, s=70
    )

    ax3d.set_xlabel("Power (W)",   fontsize=9, labelpad=8)
    ax3d.set_ylabel("Area (LUTs)", fontsize=9, labelpad=8)
    ax3d.set_zlabel("SQNR (dB)",   fontsize=9, labelpad=8)
    ax3d.set_title(
        f"FFT-{fft_size}  |  {status_label}\n"
        f"Colour = Critical Path Delay (ns)  |  "
        f"Clock target = {REFERENCE_CLOCK_PERIOD_NS:.1f} ns",
        fontsize=11
    )
    cbar = fig3d.colorbar(sc, ax=ax3d, pad=0.12, shrink=0.6,
                          label='Crit Path Delay (ns)')
    cbar.ax.axhline(REFERENCE_CLOCK_PERIOD_NS,
                    color='navy', linewidth=2, linestyle='--')
    cbar.ax.text(1.35, REFERENCE_CLOCK_PERIOD_NS / clim_max,
                 f' ← {REFERENCE_CLOCK_PERIOD_NS:.0f} ns target',
                 transform=cbar.ax.transAxes, va='center', fontsize=8,
                 color='navy')

    path_3d = os.path.join(results_subdir, f"pareto_3d_fft{fft_size}.png")
    fig3d.savefig(path_3d, dpi=DPI, bbox_inches='tight')
    plt.close(fig3d)
    log_message(f"3-D Pareto plot saved → {path_3d}")

    # ── Figure 3: dedicated latency dashboard ────────────────────────────
    fig_lat, axes_lat = plt.subplots(1, 3, figsize=(18, 5))
    fig_lat.suptitle(
        f"FFT-{fft_size}  |  Critical Path Delay Analysis  |  "
        f"{status_label}  ({n} solutions)\n"
        f"Clock target: {REFERENCE_CLOCK_PERIOD_NS:.1f} ns  "
        f"|  Timing pass rate: {pct_ok:.0f}%",
        fontsize=12, fontweight='bold'
    )

    # Panel A: histogram of critical path delay
    ax_hist = axes_lat[0]
    bins    = min(20, max(5, n // 3))
    ok_vals  = crit_delay[meets_timing.astype(bool)]
    bad_vals = crit_delay[~meets_timing.astype(bool)]
    if len(ok_vals):
        ax_hist.hist(ok_vals,  bins=bins, color=_TIMING_OK_COLOR,
                     alpha=0.75, label='Meets timing', edgecolor='k', linewidth=0.4)
    if len(bad_vals):
        ax_hist.hist(bad_vals, bins=bins, color=_TIMING_BAD_COLOR,
                     alpha=0.75, label='Violates timing', edgecolor='k', linewidth=0.4)
    ax_hist.axvline(REFERENCE_CLOCK_PERIOD_NS, color='navy',
                    linestyle='--', linewidth=1.5,
                    label=f'Target = {REFERENCE_CLOCK_PERIOD_NS:.1f} ns')
    ax_hist.set_xlabel("Critical Path Delay (ns)", fontsize=10)
    ax_hist.set_ylabel("Count",                    fontsize=10)
    ax_hist.set_title("Delay Distribution",         fontsize=11)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, linestyle='--', alpha=0.4)

    # Panel B: norm latency vs SQNR
    ax_nlat = axes_lat[1]
    _scatter_with_timing(
        ax_nlat, sqnr, norm_latency, meets_timing,
        xlabel="SQNR (dB)",
        ylabel=f"Norm Latency  (×{REFERENCE_CLOCK_PERIOD_NS:.0f} ns clock)",
        title="Norm Latency vs SQNR"
    )
    ax_nlat.axhline(1.0, color='navy', linestyle='--', linewidth=1.2,
                    label='Timing budget = 1.0')
    ax_nlat.legend(fontsize=8)

    # Panel C: crit delay vs power (scatter sized by area)
    ax_cpd = axes_lat[2]
    area_norm = (area - area.min()) / (area.max() - area.min() + 1e-9)
    sizes_cpd = 30 + 200 * area_norm        # bubble area proportional to LUT count

    sc_cpd = ax_cpd.scatter(
        power, crit_delay,
        c=np.where(meets_timing, _TIMING_OK_COLOR, _TIMING_BAD_COLOR),
        s=sizes_cpd, alpha=0.80, edgecolors='k', linewidths=0.5
    )
    ax_cpd.axhline(REFERENCE_CLOCK_PERIOD_NS, color='navy',
                   linestyle='--', linewidth=1.5,
                   label=f'Target = {REFERENCE_CLOCK_PERIOD_NS:.1f} ns')
    ax_cpd.set_xlabel("Power (W)",              fontsize=10)
    ax_cpd.set_ylabel("Critical Path Delay (ns)", fontsize=10)
    ax_cpd.set_title("Crit Delay vs Power\n(bubble size ∝ Area)",
                     fontsize=11)
    ax_cpd.legend(fontsize=8)
    ax_cpd.grid(True, linestyle='--', alpha=0.4)

    # Fake legend for timing colours
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=_TIMING_OK_COLOR, markersize=9,
               label='Meets timing'),
        Line2D([0], [0], marker='X', color='w',
               markerfacecolor=_TIMING_BAD_COLOR, markersize=9,
               label='Violates timing'),
    ]
    ax_cpd.legend(handles=legend_handles, fontsize=8, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path_lat = os.path.join(results_subdir, f"pareto_latency_fft{fft_size}.png")
    fig_lat.savefig(path_lat, dpi=DPI, bbox_inches='tight')
    plt.close(fig_lat)
    log_message(f"Latency dashboard saved → {path_lat}")


# ---------------------------------------------------------------------------
# Per-FFT-size results saving
# ---------------------------------------------------------------------------

def save_optimization_results(result, callback, fft_size):
    """
    Persist all artefacts for one FFT run:
      • pareto_objectives.npy / pareto_solutions.npy
      • fitness_history.npz
      • summary.txt                         ← latency column + Best Latency section
      • all_solutions_fft{N}.csv            ← norm_latency + crit_delay_ns columns
      • pareto_2d_fft{N}.png               ← 6-panel pairwise (timing-colour-coded)
      • pareto_3d_fft{N}.png               ← 3-D Power×Area×SQNR, colour=CritDelay
      • pareto_latency_fft{N}.png          ← dedicated latency dashboard
      • solution_logs_fft{N}.zip
      • rtl_fft{N}.zip
    """
    log_message("Saving optimization results...")

    results_subdir = os.path.join(RESULTS_DIR, f"fft_{fft_size}")
    os.makedirs(results_subdir, exist_ok=True)

    # ── Determine Pareto front (or least-infeasible fallback) ────────────
    pareto_objectives = result.F
    pareto_solutions  = result.X
    feasible = pareto_solutions is not None

    if not feasible:
        log_message(
            "WARNING: No feasible solutions — saving least-infeasible fallback.",
            level='WARN'
        )
        pop = result.pop
        if pop is not None and len(pop) > 0:
            pareto_objectives = pop.get("F")
            pareto_solutions  = pop.get("X")
            cv_vals           = pop.get("CV")
            if cv_vals is not None:
                order             = np.argsort(cv_vals.ravel())
                pareto_objectives = pareto_objectives[order]
                pareto_solutions  = pareto_solutions[order]
        else:
            pareto_objectives = np.empty((0, OBJECTIVES))
            pareto_solutions  = np.empty((0,), dtype=int)

    np.save(os.path.join(results_subdir, 'pareto_objectives.npy'), pareto_objectives)
    np.save(os.path.join(results_subdir, 'pareto_solutions.npy'),  pareto_solutions)
    np.savez(os.path.join(results_subdir, 'fitness_history.npz'), *callback.data)

    # ── Summary text ─────────────────────────────────────────────────────
    front_label  = "Pareto" if feasible else "Fallback"
    summary_file = os.path.join(results_subdir, 'summary.txt')

    with open(summary_file, 'w') as f:
        f.write("Mixed-Precision FFT Optimization Results\n")
        f.write(f"{'='*72}\n\n")
        f.write(f"FFT Size              : {fft_size}\n")
        f.write(f"Population            : {POPULATION}\n")
        f.write(f"Generations           : {GENERATIONS}\n")
        f.write(f"Objectives            : {OBJECTIVES}  "
                f"(Power, Area, SQNR, Critical-Path Delay)\n")
        f.write(f"Clock target          : {REFERENCE_CLOCK_PERIOD_NS:.1f} ns\n")
        f.write(f"FPGA device           : {FPGA_DEVICE}\n\n")

        if not feasible:
            f.write("*** WARNING: No feasible solutions found. ***\n")
            f.write("Showing least-infeasible solutions from the final population.\n\n")

        n_sol = len(pareto_solutions)
        f.write(f"{front_label} Front Solutions: {n_sol}\n\n")

        if n_sol == 0:
            f.write("No solutions to report.\n")
        else:
            # ── Decode all solutions ──────────────────────────────────────
            decoded = [_decode_objectives(pareto_objectives[i], fft_size)
                       for i in range(n_sol)]

            n_timing_ok = sum(1 for d in decoded if d['meets_timing'])
            f.write(f"Timing pass rate      : {n_timing_ok}/{n_sol} "
                    f"({100*n_timing_ok/n_sol:.0f}%)\n\n")

            # ── Table header ──────────────────────────────────────────────
            hdr = (f"{'ID':<5} {'Power(W)':<12} {'Area(LUTs)':<12} "
                   f"{'SQNR(dB)':<12} {'NormLat':<10} "
                   f"{'CritDelay(ns)':<15} {'MeetsTiming':<12}")
            f.write(hdr + "\n")
            f.write('-' * len(hdr) + '\n')

            for i, d in enumerate(decoded):
                sqnr_str = (f"{d['sqnr_db']:.2f}"
                            if not math.isinf(d['sqnr_db']) else "  inf")
                crit_str = f"{d['crit_delay_ns']:.3f}"
                if d['norm_latency'] >= 10.0:
                    crit_str = f">={crit_str}"   # clamped value
                timing_str = "YES" if d['meets_timing'] else "NO "
                f.write(
                    f"{i:<5} {d['power_W']:<12.6f} {int(d['area_luts']):<12} "
                    f"{sqnr_str:<12} {d['norm_latency']:<10.4f} "
                    f"{crit_str:<15} {timing_str:<12}\n"
                )

            # ── Best per objective ────────────────────────────────────────
            obj_arr = np.array(pareto_objectives)
            f.write("\n\nBest Solutions by Objective:\n")
            f.write('-' * 60 + '\n')

            best_specs = [
                ("Best Power (min)",       0, "power_W",      "W"),
                ("Best Area (min)",        1, "area_luts",    "LUTs"),
                ("Best SQNR (max perf)",   2, "sqnr_db",      "dB"),
                ("Best Crit-Path (min)",   3, "norm_latency", "norm"),
            ]

            for label, col, key, unit in best_specs:
                idx = int(np.argmin(obj_arr[:, col]))
                d   = decoded[idx]
                f.write(f"\n{label}:\n")
                f.write(f"  Solution ID       : {idx}\n")
                f.write(f"  Power             : {d['power_W']:.6f} W\n")
                f.write(f"  Area              : {int(d['area_luts'])} LUTs\n")
                sqnr_str = (f"{d['sqnr_db']:.2f} dB"
                            if not math.isinf(d['sqnr_db']) else "inf dB")
                f.write(f"  SQNR              : {sqnr_str}\n")
                f.write(f"  Norm Latency      : {d['norm_latency']:.4f}x "
                        f"(clock = {REFERENCE_CLOCK_PERIOD_NS:.1f} ns)\n")
                f.write(f"  Crit Path Delay   : {d['crit_delay_ns']:.3f} ns")
                f.write(f"  {'  ← MEETS TIMING' if d['meets_timing'] else '  ← VIOLATES TIMING'}\n")
                f.write(f"  Chromosome        : {list(pareto_solutions[idx])}\n")

    log_message(f"Summary saved → {summary_file}")

    # ── CSV of all evaluated solutions ────────────────────────────────────
    export_solutions_csv(result, fft_size, results_subdir)

    # ── Pareto front plots (4-objective, redesigned) ──────────────────────
    plot_pareto_front(pareto_objectives, fft_size, results_subdir, feasible)

    # ── Parse gen*_sol*.txt → all-generations CSV, then zip+delete them ──
    txt_files = parse_solution_txts_to_csv(fft_size, results_subdir)
    compress_solution_txt_files(fft_size, results_subdir, txt_files)

    # ── Compress RTL files ────────────────────────────────────────────────
    compress_rtl_files(results_subdir, fft_size)

    log_message(
        f"Results saved to {results_subdir}  "
        f"({front_label} front: {n_sol} solutions)"
    )


# ---------------------------------------------------------------------------
# Per-size optimisation runner
# ---------------------------------------------------------------------------

def run_optimization_for_fft_size(fft_size):
    """Run NSGA-II optimisation for a specific FFT size; returns pymoo result."""
    import globalVariablesMixedFFT
    globalVariablesMixedFFT.CURRENT_FFT_SIZE = fft_size

    log_message(f"\n{'='*60}")
    log_message(f"Starting optimisation for {fft_size}-point FFT")
    log_message(f"{'='*60}\n")

    problem  = MixedPrecisionFFTProblem(fft_size=fft_size)
    callback = MyCallback()

    algorithm = NSGA2(
        pop_size=POPULATION,
        sampling=SmartInitialSampling(),
        crossover=StagewiseCrossover(fft_size=fft_size, prob=CROSSOVER_RATE),
        mutation=BlockwiseMutation(fft_size=fft_size),
    )
    termination = get_termination("n_gen", GENERATIONS)

    log_message("NSGA-II Configuration:")
    log_message(f"  Population size : {POPULATION}")
    log_message(f"  Generations     : {GENERATIONS}")
    log_message(f"  Crossover rate  : {CROSSOVER_RATE}")
    log_message(f"  Mutation rate   : {MUTATION_RATE}")
    log_message(f"  Objectives      : {OBJECTIVES}  (Power, Area, SQNR, CritDelay)")
    log_message(f"  Parallel threads: {SOLUTION_THREADS}")
    log_message(f"  Clock target    : {REFERENCE_CLOCK_PERIOD_NS:.1f} ns")

    result = minimize(
        problem,
        algorithm,
        termination,
        save_history=False,
        callback=callback,
        seed=SEED,
        verbose=VERBOSE,
    )

    log_message(f"Optimisation complete for {fft_size}-point FFT")
    save_optimization_results(result, callback, fft_size)
    return result


# ---------------------------------------------------------------------------
# Full sweep + cross-FFT summary outputs
# ---------------------------------------------------------------------------

def generate_comprehensive_summary(all_results):
    """
    After the full sweep write:
      results/comprehensive_summary.txt     ← latency range + Min Latency row
      results/all_pareto_solutions.csv      ← norm_latency + crit_delay_ns columns
      results/comparison_all_fft_sizes.png  ← 4-metric sweep plot
    """
    # ── Text summary ──────────────────────────────────────────────────────
    summary_file = os.path.join(RESULTS_DIR, 'comprehensive_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Mixed-Precision FFT Optimization — Comprehensive Summary\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Clock target: {REFERENCE_CLOCK_PERIOD_NS:.1f} ns  |  "
                f"FPGA: {FPGA_DEVICE}\n\n")

        for fft_size, result in sorted(all_results.items()):
            f.write(f"\nFFT Size: {fft_size}\n")
            f.write("-" * 72 + "\n")
            if result is None:
                f.write("  Optimisation failed\n")
                continue

            pf = result.F if result.F is not None else np.empty((0, OBJECTIVES))
            n  = len(pf)
            f.write(f"  Pareto front size  : {n}\n")

            if n == 0:
                continue

            decoded = [_decode_objectives(pf[i], fft_size) for i in range(n)]

            powers  = np.array([d['power_W']      for d in decoded])
            areas   = np.array([d['area_luts']     for d in decoded])
            sqnrs   = np.array([d['sqnr_db']       for d in decoded
                                if not math.isinf(d['sqnr_db'])])
            delays  = np.array([d['crit_delay_ns'] for d in decoded])
            n_ok    = sum(1 for d in decoded if d['meets_timing'])

            f.write(f"  Power range        : {powers.min():.6f} – {powers.max():.6f} W\n")
            f.write(f"  Area range         : {areas.min():.0f} – {areas.max():.0f} LUTs\n")
            if len(sqnrs):
                f.write(f"  SQNR range         : {sqnrs.min():.2f} – {sqnrs.max():.2f} dB\n")
            f.write(f"  Crit-path range    : {delays.min():.3f} – {delays.max():.3f} ns\n")
            f.write(f"  Timing pass rate   : {n_ok}/{n} ({100*n_ok/n:.0f}%)\n")

    log_message(f"Comprehensive summary → {summary_file}")

    # ── Combined CSV ───────────────────────────────────────────────────────
    combined_csv = os.path.join(RESULTS_DIR, 'all_pareto_solutions.csv')
    with open(combined_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fft_size', 'solution_id',
                         'power_W', 'area_LUTs', 'sqnr_dB',
                         'norm_latency', 'crit_delay_ns', 'meets_timing'])
        for fft_size, result in sorted(all_results.items()):
            if result is None or result.F is None:
                continue
            for i, obj in enumerate(result.F):
                d = _decode_objectives(obj, fft_size)
                sqnr_str = (f"{d['sqnr_db']:.4f}"
                            if not math.isinf(d['sqnr_db']) else "inf")
                writer.writerow([
                    fft_size, i,
                    f"{d['power_W']:.6f}",
                    int(d['area_luts']),
                    sqnr_str,
                    f"{d['norm_latency']:.4f}",
                    f"{d['crit_delay_ns']:.3f}",
                    int(d['meets_timing']),
                ])
    log_message(f"Combined Pareto CSV → {combined_csv}")

    # ── 4-metric comparison plot ───────────────────────────────────────────
    sizes      = []
    best_power, best_area, best_sqnr, best_delay = [], [], [], []

    for fft_size, result in sorted(all_results.items()):
        if result is None or result.F is None or len(result.F) == 0:
            continue
        pf      = result.F
        decoded = [_decode_objectives(pf[i], fft_size) for i in range(len(pf))]

        powers  = [d['power_W']      for d in decoded]
        areas   = [d['area_luts']    for d in decoded]
        delays  = [d['crit_delay_ns'] for d in decoded]
        sqnrs_f = [d['sqnr_db']      for d in decoded
                   if not math.isinf(d['sqnr_db']) and not math.isnan(d['sqnr_db'])]

        sizes.append(fft_size)
        best_power.append(min(powers))
        best_area.append(min(areas))
        best_delay.append(min(delays))
        best_sqnr.append(max(sqnrs_f) if sqnrs_f else 0.0)

    if sizes:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            "Best Achievable Metrics vs FFT Size\n"
            f"(Clock target = {REFERENCE_CLOCK_PERIOD_NS:.1f} ns  |  "
            f"FPGA: {FPGA_DEVICE})",
            fontsize=13, fontweight='bold'
        )

        metrics = [
            (best_power, "Min Power (W)",           _OBJ_COLORS['power'],   axes[0, 0], False),
            (best_area,  "Min Area (LUTs)",          _OBJ_COLORS['area'],    axes[0, 1], False),
            (best_sqnr,  "Max SQNR (dB)",            _OBJ_COLORS['sqnr'],    axes[1, 0], False),
            (best_delay, "Min Crit Path Delay (ns)", _OBJ_COLORS['latency'], axes[1, 1], True),
        ]

        for ydata, ylabel, color, ax, add_ref in metrics:
            ax.plot(sizes, ydata, 'o-', color=color, linewidth=2,
                    markersize=8, markeredgecolor='k', markeredgewidth=0.6)
            ax.set_xlabel("FFT Size (points)", fontsize=10)
            ax.set_ylabel(ylabel,              fontsize=10)
            ax.set_title(ylabel,               fontsize=11)
            ax.set_xscale('log', base=2)
            ax.set_xticks(sizes)
            ax.set_xticklabels([str(s) for s in sizes], rotation=45, ha='right')
            ax.grid(True, linestyle='--', alpha=0.4)

            if add_ref:
                ax.axhline(REFERENCE_CLOCK_PERIOD_NS, color='navy',
                           linestyle='--', linewidth=1.5,
                           label=f'Clock = {REFERENCE_CLOCK_PERIOD_NS:.1f} ns')
                ax.legend(fontsize=9)

            # Annotate each data point
            for sx, sy in zip(sizes, ydata):
                ax.annotate(
                    f"{sy:.3g}", (sx, sy),
                    textcoords="offset points", xytext=(0, 7),
                    ha='center', fontsize=8, color=color
                )

        plt.tight_layout()
        comp_plot = os.path.join(RESULTS_DIR, 'comparison_all_fft_sizes.png')
        fig.savefig(comp_plot, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        log_message(f"4-metric comparison plot → {comp_plot}")


def run_full_optimization_sweep():
    """Run optimisation for every FFT size in FFT_SIZES (2 – 1024)."""
    log_message("\n" + "=" * 60)
    log_message("Mixed-Precision FFT Optimization Framework")
    log_message("=" * 60 + "\n")

    setup_verilog_sources()
    all_results = {}

    for fft_size in FFT_SIZES:
        try:
            global CURRENT_GEN
            CURRENT_GEN = 0
            result = run_optimization_for_fft_size(fft_size)
            all_results[fft_size] = result
        except Exception as e:
            log_message(
                f"ERROR: Optimisation failed for {fft_size}-point FFT: {e}",
                level='ERROR'
            )
            all_results[fft_size] = None

    generate_comprehensive_summary(all_results)

    log_message("\n" + "=" * 60)
    log_message("Optimisation sweep complete!")
    log_message("=" * 60)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

def quick_test():
    """Quick smoke-test: 16-point FFT with reduced pop/gen."""
    log_message("Running quick test with 16-point FFT")
    setup_verilog_sources()

    global CURRENT_GEN, POPULATION, GENERATIONS
    CURRENT_GEN = 0
    orig_pop, orig_gen = POPULATION, GENERATIONS
    POPULATION, GENERATIONS = 6, 3

    run_optimization_for_fft_size(fft_size=16)

    POPULATION, GENERATIONS = orig_pop, orig_gen
    log_message("Quick test complete")


# ---------------------------------------------------------------------------
# Entry-points
# ---------------------------------------------------------------------------

def main():
    """
    Default entry-point: full sweep across all FFT sizes (2 → 1024).
    """
    run_full_optimization_sweep()
    # quick_test()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Mixed-Precision FFT Optimization using NSGA-II'
    )
    parser.add_argument(
        '--mode',
        choices=['test', 'single', 'full'],
        default='full',
        help=(
            'test   – quick 16-pt smoke test | '
            'single – one FFT size (--fft-size) | '
            'full   – complete sweep 2→1024 (default)'
        ),
    )
    parser.add_argument(
        '--fft-size',
        type=int,
        default=8,
        choices=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        help='FFT size for --mode single',
    )

    args = parser.parse_args()

    if args.mode == 'test':
        quick_test()
    elif args.mode == 'single':
        setup_verilog_sources()
        run_optimization_for_fft_size(args.fft_size)
    else:
        main()
