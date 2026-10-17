"""
Objective Evaluation for Mixed-Precision FFT Optimization
Uses ACTUAL Vivado synthesis critical path delay as 4th objective (on-chip latency).
"""

import numpy as np
import subprocess
import os
import csv
import hashlib
import math
from pymoo.core.problem import Problem
from concurrent.futures import ThreadPoolExecutor, as_completed

from globalVariablesMixedFFT import *
from fft_template_generator import FFTTemplateGenerator
from performance_evaluator import PerformanceEvaluator


class MixedPrecisionFFTProblem(Problem):
    def __init__(self, fft_size=8, **kwargs):
        self.fft_size     = fft_size
        self.template_gen = FFTTemplateGenerator(fft_size)
        self.perf_eval    = PerformanceEvaluator(fft_size)

        chrom_length = self.template_gen.get_chromosome_length()

        super().__init__(
            n_var=chrom_length,
            n_obj=OBJECTIVES,           # 4 objectives
            n_ieq_constr=3,
            xl=[0] * chrom_length,
            xu=[1] * chrom_length,
            vtype=int,
            elementwise_evaluation=False,
            **kwargs
        )

        log_message(f"Initialized FFT-{fft_size} problem with Vivado timing as 4th objective")

    def _evaluate(self, X, out, *args, **kwargs):
        global CURRENT_GEN
        log_message(f"=== Generation {CURRENT_GEN} ===", level='GEN')
        with open('generation.txt', 'w') as f:
            f.write(str(CURRENT_GEN))
        CURRENT_GEN += 1

        F = [None] * len(X)
        G = [None] * len(X)

        with ThreadPoolExecutor(max_workers=SOLUTION_THREADS) as executor:
            futures = {executor.submit(self.evaluate_solution, X[i], i): i for i in range(len(X))}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    f_vals, g_vals = future.result()
                    F[idx] = f_vals
                    G[idx] = g_vals
                except Exception as e:
                    log_message(f"Solution {idx} failed: {e}", level='ERROR')
                    F[idx] = [MAX_POWER_W*2, MAX_AREA_LUTS*2, 1e6, 50.0]  # heavy latency penalty
                    G[idx] = [MAX_POWER_W, MAX_AREA_LUTS, MIN_SQNR_DB]

        out["F"] = np.array(F)
        out["G"] = np.array(G)
        log_message(f"Generation {CURRENT_GEN-1} complete")

    def evaluate_solution(self, chromosome, sol_id):
        log_message(f"Evaluating solution {sol_id}: {list(chromosome)}")

        chrom_hash = self._hash_chromosome(chromosome)
        if ENABLE_RESULT_CACHE and chrom_hash in RESULT_CACHE:
            return self._compute_objectives_and_constraints(RESULT_CACHE[chrom_hash])

        design_name = f"fft_{self.fft_size}_sol{sol_id}_gen{CURRENT_GEN}"

        core_file = os.path.join(GENERATED_DESIGNS_DIR, f"{design_name}.v")
        core_file, top_file = self.template_gen.generate_verilog(chromosome, core_file)

        power, area, crit_delay = self._run_vivado_synthesis(design_name, core_file, top_file)
        sqnr = self._run_performance_evaluation(core_file, design_name, chromosome)

        norm_latency = self._compute_actual_normalized_latency(crit_delay)

        results = {
            'power': power,
            'area': area,
            'sqnr': sqnr,
            'norm_latency': norm_latency,
            'crit_delay_ns': crit_delay
        }

        RESULT_CACHE[chrom_hash] = results
        self._save_solution_result(sol_id, chromosome, results)

        stats = self.template_gen.analyze_chromosome_statistics(chromosome)
        log_message(f"Solution {sol_id}: P={power:.4f}W, A={area} LUTs, SQNR={sqnr:.2f}dB, CritDelay={crit_delay:.3f}ns → NormLat={norm_latency:.3f}x")

        return self._compute_objectives_and_constraints(results)

    def _hash_chromosome(self, chromosome):
        return hashlib.md5(''.join(map(str, chromosome)).encode()).hexdigest()

    def _run_vivado_synthesis(self, design_name, core_file, top_file):
        log_message(f"Running Vivado synthesis for {design_name}")

        csv_output = os.path.join(REPORTS_DIR, f"{design_name}_metrics.csv")
        verilog_dir = os.path.abspath(VERILOG_SOURCES_DIR)
        core_abs = os.path.abspath(core_file)
        top_abs = os.path.abspath(top_file)

        cmd = [
            VIVADO_PATH, '-mode', 'batch', '-source', './vivado_synthesis.tcl',
            '-tclargs', design_name, csv_output, str(CLOCK_PERIOD),
            core_abs, top_abs, verilog_dir
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                log_message(f"Vivado failed for {design_name}", level='ERROR')
                return MAX_POWER_W*2, MAX_AREA_LUTS*2, 200.0
            return self._parse_vivado_metrics(csv_output)
        except Exception as e:
            log_message(f"Vivado error: {e}", level='ERROR')
            return MAX_POWER_W*2, MAX_AREA_LUTS*2, 200.0

    def _parse_vivado_metrics(self, csv_file):
        power = MAX_POWER_W * 2
        area = MAX_AREA_LUTS * 2
        crit_delay = 200.0

        try:
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Metric'] == 'total_power_w':
                        power = float(row['Value'])
                    elif row['Metric'] == 'lut_count':
                        area = int(row['Value'])
                    elif row['Metric'] == 'critical_path_delay_ns':
                        crit_delay = float(row['Value'])
        except Exception as e:
            log_message(f"Error parsing metrics from {csv_file}: {e}", level='ERROR')

        return power, area, crit_delay

    def _compute_actual_normalized_latency(self, crit_delay_ns):
        """Normalize actual critical path delay from Vivado synthesis."""
        if crit_delay_ns <= 0 or math.isnan(crit_delay_ns) or math.isinf(crit_delay_ns):
            return 10.0

        # Basic normalization against target clock
        norm = crit_delay_ns / REFERENCE_CLOCK_PERIOD_NS

        # Rough scaling with number of stages (pipeline depth)
        num_stages = self.template_gen.num_stages
        pipeline_factor = max(1.0, num_stages / 6.0)

        return min(norm * pipeline_factor, 10.0)

    def _run_performance_evaluation(self, verilog_file, design_name, chromosome=None):
        try:
            return self.perf_eval.evaluate_design(verilog_file, design_name, chromosome=chromosome)
        except Exception as e:
            log_message(f"Performance evaluation failed: {e}", level='ERROR')
            return -100.0

    def _compute_objectives_and_constraints(self, results):
        power = results['power']
        area = results['area']
        sqnr = results['sqnr']
        norm_latency = results.get('norm_latency', 10.0)

        sqnr_clamped = max(sqnr, 0.0)
        perf_error = 1.0 / (sqnr_clamped + 1.0)

        objectives = [
            power * WEIGHT_POWER,
            area * WEIGHT_AREA,
            perf_error * WEIGHT_PERFORMANCE,
            norm_latency * WEIGHT_LATENCY      # 4th objective: actual Vivado timing
        ]

        constraints = [
            power - MAX_POWER_W,
            area - MAX_AREA_LUTS,
            MIN_SQNR_DB - sqnr
        ]
        return objectives, constraints

    def _save_solution_result(self, sol_id, chromosome, results):
        result_file = os.path.join(RESULTS_DIR, f"gen{CURRENT_GEN}_sol{sol_id}.txt")
        stats = self.template_gen.analyze_chromosome_statistics(chromosome)

        with open(result_file, 'w') as f:
            f.write(f"FFT Size          : {self.fft_size}\n")
            f.write(f"Generation        : {CURRENT_GEN}\n")
            f.write(f"Solution ID       : {sol_id}\n")
            f.write(f"Chromosome        : {list(chromosome)}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Power             : {results['power']:.6f} W\n")
            f.write(f"  Area              : {results['area']} LUTs\n")
            f.write(f"  SQNR              : {results['sqnr']:.2f} dB\n")
            f.write(f"  Crit Path Delay   : {results.get('crit_delay_ns', 0):.3f} ns\n")
            f.write(f"  Norm Latency      : {results.get('norm_latency', 0):.4f}x\n")
            f.write(f"\nPrecision Stats:\n")
            for k, v in stats.items():
                if not isinstance(v, list):
                    f.write(f"  {k}: {v}\n")

# Quick test
if __name__ == "__main__":
    problem = MixedPrecisionFFTProblem(fft_size=8)
    test_chrom = np.array([0,0,1,0,1,1,0,0])
    objectives, constraints = problem.evaluate_solution(test_chrom, 0)
    print(f"4 Objectives: {objectives}")
    print(f"Constraints : {constraints}")
