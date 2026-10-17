"""
Performance Evaluation Module
==============================
Calculates SQNR (Signal-to-Quantisation-Noise Ratio) by running
iverilog/vvp simulation of generated mixed-precision FFT designs and
comparing against an FP32 NumPy reference.


    SQNR = 10 * log10( signal_power / noise_power )

where
    signal_power = mean( |golden|^2 )
    noise_power  = mean( |golden - approximate|^2 )

This is reported per test signal and averaged across all signals.

Testbench interface matches the generated top module:
  - load_en / load_addr / load_data   (input loading, FP8 16-bit)
  - start / done                      (FFT trigger / completion)
  - unload_en / unload_addr / unload_data (result readout, 16-bit)

Test signals
------------
  1. Impulse              – all energy in DC bin; deterministic worst-case
  2. Single Tone          – one real sinusoid; narrow-band stress test
  3. Multi-Tone           – several tones; moderate bandwidth
  4. Chirp (LFM)          – linear frequency sweep; broadband, envelope ≈1
  5. Sinusoid (complex)   – full-scale complex exponential at Nyquist/4
  6. White Noise          – broadband random; statistical average behaviour
  7. Step Function        – time-domain discontinuity; tests Gibbs artefacts
  8. Gaussian Pulse       – smooth broadband; tests small out-of-band values

Key design decisions
--------------------
* Active-low async reset: tb drives rst=0 for reset, rst=1 to release.
* 2-cycle memory read latency: unload loop waits 3 @posedge per sample.
* Input packed as FP8: load_data = {fp8_real[7:0], fp8_imag[7:0]}.
* Output parsed as FP8 or FP4 depending on final stage precision.
* One twiddles file is generated per simulation run in ./sim/.
* Golden reference is computed from FP8-quantised inputs (same as HW).
"""

import numpy as np
import subprocess
import os
import glob as glob_module
import math


# ---------------------------------------------------------------------------
# Human-readable labels for each test vector (must stay in sync with
# _generate_test_vectors ordering).
# ---------------------------------------------------------------------------
SIGNAL_LABELS = [
    "Impulse",
    "Single Tone",
    "Multi-Tone",
    "Chirp (LFM)",
    "Sinusoid (complex)",
    # "White Noise",
    "Step Function",
    "Gaussian Pulse",
    # Radar-specific signals
    "Radar Pulsed Sinusoid",
    "Radar Clutter + Target",
    "Radar Barker-13 Pulse",
    "Radar Doppler Burst",
]


class PerformanceEvaluator:
    def __init__(self, fft_size):
        self.fft_size            = fft_size
        self.num_stages          = int(math.log2(fft_size))
        self.verilog_sources_dir = './verilog_sources'
        self.test_vectors        = self._generate_test_vectors()
        self.golden_outputs      = self._compute_golden_outputs()

    # ==================================================================
    # Test vectors
    # ==================================================================
    def _generate_test_vectors(self):
        """
        Generate a diverse suite of test vectors that stress different
        aspects of FFT arithmetic precision.

        All vectors are normalised so that max |sample| <= 0.9, keeping
        values comfortably within the FP8 E4M3 dynamic range.

        Ordering must match SIGNAL_LABELS above.
        """
        n     = self.fft_size
        n_arr = np.arange(n, dtype=float)
        vecs  = []

        # ------------------------------------------------------------------
        # 1. Impulse at index 0 — FFT = constant 1+0j across all bins.
        #    Classic deterministic test; very low SQNR reveals rounding in
        #    twiddle multiplications (every output must equal 1).
        # ------------------------------------------------------------------
        v = np.zeros(n, dtype=complex)
        v[0] = 0.9
        vecs.append(v)

        # ------------------------------------------------------------------
        # 2. Single tone — real sinusoid at bin k.
        #    Energy concentrated in two bins (±k); tests narrow-band fidelity.
        # ------------------------------------------------------------------
        k = max(1, n // 8)          # bin well away from DC
        v = 0.9 * np.cos(2 * np.pi * k * n_arr / n).astype(complex)
        vecs.append(v)

        # ------------------------------------------------------------------
        # 3. Multi-tone — sum of several real sinusoids.
        #    Moderate bandwidth; amplitude controlled to avoid clipping.
        # ------------------------------------------------------------------
        tones = [1, 3, 5, 7] if n >= 8 else [1]
        v = np.zeros(n, dtype=complex)
        for kt in tones:
            if kt < n // 2:
                v += np.exp(2j * np.pi * kt * n_arr / n)
        peak = np.max(np.abs(v))
        v = v / peak * 0.9 if peak > 0 else v
        vecs.append(v)

        # ------------------------------------------------------------------
        # 4. Chirp (Linear Frequency Modulation, LFM).
        #    Phase sweeps quadratically → energy spread across all bins.
        #    |v[n]| = 1 exactly, so no clipping.
        # ------------------------------------------------------------------
        v = np.exp(1j * np.pi * n_arr ** 2 / n) * 0.9
        vecs.append(v)

        # ------------------------------------------------------------------
        # 5. Sinusoid — complex exponential at Nyquist/4 (bin = N/4).
        #    Full-scale complex; exercises both real and imaginary paths.
        # ------------------------------------------------------------------
        k5 = max(1, n // 4)
        v  = 0.9 * np.exp(2j * np.pi * k5 * n_arr / n)
        vecs.append(v)

        # ------------------------------------------------------------------
        # 6. White noise — complex Gaussian, fixed seed for reproducibility.
        #    Broadband; SQNR averaged over all bins = statistical behaviour.
        # ------------------------------------------------------------------
        # rng = np.random.default_rng(42)
        # v   = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        # peak = np.max(np.abs(v))
        # v = v / peak * 0.9 if peak > 0 else v
        # vecs.append(v)

        # ------------------------------------------------------------------
        # 7. Step function — first half +1, second half -1 (real).
        #    Discontinuity excites all odd harmonics; tests Gibbs artefacts.
        # ------------------------------------------------------------------
        v = np.zeros(n, dtype=complex)
        v[:n // 2] =  0.9
        v[n // 2:] = -0.9
        vecs.append(v)

        # ------------------------------------------------------------------
        # 8. Gaussian pulse — real, centred in time domain.
        #    Smooth bell shape; tests small out-of-band spectral values where
        #    quantisation noise is most visible relative to signal.
        # ------------------------------------------------------------------
        sigma  = n / 8.0
        centre = n / 2.0
        v = np.exp(-0.5 * ((n_arr - centre) / sigma) ** 2).astype(complex)
        peak = np.max(np.abs(v))
        v = v / peak * 0.9 if peak > 0 else v
        vecs.append(v)

        # ------------------------------------------------------------------
        # 9. Radar Pulsed Sinusoid — a gated CW burst simulating a radar
        #    return pulse.  A rectangular window of duration n/4 centred in
        #    the block is modulated by a carrier at bin k_radar.  This
        #    stresses the sidelobe structure of the rectangular window and
        #    reveals Gibbs-like artefacts in the spectral domain that low-
        #    precision arithmetic amplifies.
        # ------------------------------------------------------------------
        k_radar = max(1, n // 6)
        pulse_start = n // 4
        pulse_end   = 3 * n // 4
        v = np.zeros(n, dtype=complex)
        v[pulse_start:pulse_end] = np.exp(
            2j * np.pi * k_radar * n_arr[pulse_start:pulse_end] / n
        )
        peak = np.max(np.abs(v))
        v = v / peak * 0.9 if peak > 0 else v
        vecs.append(v)

        # ------------------------------------------------------------------
        # 10. Radar Clutter + Target — strong clutter component (DC-adjacent
        #     bins) plus a weak target tone at a different bin.  The dynamic
        #     range challenge (strong clutter swamping the target) is the
        #     canonical radar FFT stress test; low-precision rounding in
        #     twiddle factors directly degrades target detectability.
        # ------------------------------------------------------------------
        k_clutter = max(1, n // 16)          # near-DC clutter
        k_target  = max(k_clutter + 2, n // 5)  # target well-separated
        clutter_amp = 0.85
        target_amp  = 0.05                   # ~24 dB below clutter
        v = (
            clutter_amp * np.exp(2j * np.pi * k_clutter * n_arr / n)
            + target_amp * np.exp(2j * np.pi * k_target  * n_arr / n)
        )
        peak = np.max(np.abs(v))
        v = v / peak * 0.9 if peak > 0 else v
        vecs.append(v)

        # ------------------------------------------------------------------
        # 11. Radar Barker-13 Pulse — a length-13 Barker-coded phase
        #     sequence (±1) zero-padded to n.  Barker codes are the
        #     canonical pulse-compression waveform; their nearly flat
        #     sidelobe spectrum probes uniform spectral coverage and
        #     rounding behaviour across all bins.
        # ------------------------------------------------------------------
        barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1],
                            dtype=float)
        v = np.zeros(n, dtype=complex)
        code_len = min(13, n)
        v[:code_len] = barker13[:code_len]
        peak = np.max(np.abs(v))
        v = v / peak * 0.9 if peak > 0 else v
        vecs.append(v)

        # ------------------------------------------------------------------
        # 12. Radar Doppler Burst — a multi-pulse Doppler scenario encoded
        #     as a complex baseband burst: the phase rotates linearly across
        #     the n samples to simulate a target moving at a fixed radial
        #     velocity.  Unlike the pure sinusoid, this uses a Hamming
        #     window envelope to suppress range sidelobes, exercising the
        #     interaction between windowing and low-precision butterfly maths.
        # ------------------------------------------------------------------
        k_doppler = max(1, n // 7)           # Doppler frequency bin
        hamming   = np.hamming(n)
        v = hamming * np.exp(2j * np.pi * k_doppler * n_arr / n)
        peak = np.max(np.abs(v))
        v = v / peak * 0.9 if peak > 0 else v
        vecs.append(v)

        assert len(vecs) == len(SIGNAL_LABELS), (
            f"Signal count mismatch: {len(vecs)} vectors vs "
            f"{len(SIGNAL_LABELS)} labels"
        )
        return vecs

    def _compute_golden_outputs(self):
        """
        Compute golden FFT outputs using FP8-quantised inputs.

        The hardware receives FP8-quantised inputs, not the original float64
        values. The golden reference must start from the same quantised values
        so that SQNR measures only FFT arithmetic error, not input quantisation
        error.

        Flow:
            float64 input
                -> quantise to FP8   (float_to_fp8_e4m3)
                -> dequantise to float64  (fp8_to_float)
                -> np.fft.fft()
                -> golden output
        """
        goldens = []
        for v in self.test_vectors:
            v_quantized = np.array([
                self.fp8_to_float(self.float_to_fp8_e4m3(x.real)) +
                1j * self.fp8_to_float(self.float_to_fp8_e4m3(x.imag))
                for x in v
            ])
            goldens.append(np.fft.fft(v_quantized))
        return goldens

    # ==================================================================
    # Float <-> FP conversion helpers
    # ==================================================================
    def float_to_fp8_e4m3(self, val):
        """Quantise a float to FP8 E4M3 (bias=7); returns 8-bit integer."""
        if val == 0.0:
            return 0
        sign_bit = 0x80 if val < 0 else 0x00
        val = abs(val)
        if val < 2 ** (-6):
            return sign_bit
        exp_u = math.floor(math.log2(val))
        exp_b = exp_u + 7
        if exp_b < 1:
            exp_b = 0
            mant  = min(7, round(val / (2 ** -6) * 8))
        elif exp_b >= 15:
            return sign_bit | 0x7E
        else:
            mant_f = val / (2 ** exp_u) - 1.0
            mant   = min(7, round(mant_f * 8))
            if mant >= 8:
                mant = 0
                exp_b += 1
                if exp_b >= 15:
                    return sign_bit | 0x7E
        return (sign_bit & 0x80) | ((exp_b & 0x0F) << 3) | (mant & 0x07)

    def fp8_to_float(self, fp8_val):
        """FP8 E4M3 (bias=7) -> float."""
        fp8_val &= 0xFF
        if fp8_val == 0:
            return 0.0
        sign = (fp8_val >> 7) & 0x1
        exp  = (fp8_val >> 3) & 0xF
        mant =  fp8_val       & 0x7
        if exp == 0:
            value = mant / 8.0 * (2 ** -6)
        elif exp == 15:
            return float('nan')
        else:
            value = (1.0 + mant / 8.0) * (2 ** (exp - 7))
        return -value if sign else value

    def float_to_fp4(self, val):
        """Quantise a float to FP4 E2M1 (bias=1); returns 4-bit integer."""
        if val == 0.0:
            return 0
        sign = 0x8 if val < 0 else 0x0
        val  = abs(val)
        if val < 0.5:
            exp = 0; mant = 1 if val >= 0.25 else 0
        elif val < 1.0:
            exp = 1; mant = 0
        elif val < 1.5:
            exp = 1; mant = 1
        elif val < 2.0:
            exp = 2; mant = 0
        elif val < 3.0:
            exp = 2; mant = 1
        else:
            exp = 3; mant = 1
        return (sign & 0x8) | ((exp & 0x3) << 1) | (mant & 0x1)

    def fp4_to_float(self, fp4_val):
        """FP4 E2M1 (bias=1) -> float."""
        fp4_val &= 0xF
        if fp4_val == 0:
            return 0.0
        sign = (fp4_val >> 3) & 0x1
        exp  = (fp4_val >> 1) & 0x3
        mant =  fp4_val       & 0x1
        if exp == 0:
            value = mant * 0.5
        else:
            value = (1.0 + mant * 0.5) * (2 ** (exp - 1))
        return -value if sign else value

    # ==================================================================
    # Twiddle file generator
    # ==================================================================
    def _write_twiddle_file(self, sim_dir):
        """
        Write twiddles_1024.txt in binary format for $readmemb.
        Format per line: 24 binary digits  (MSB first)
          [23:16] FP8 real
          [15:8]  FP8 imag
          [7:4]   FP4 real
          [3:0]   FP4 imag
        Only the first 512 entries are used (symmetry handles the rest).
        """
        path = os.path.join(sim_dir, 'twiddles_1024.txt')
        with open(path, 'w') as f:
            for idx in range(512):
                angle = -2.0 * math.pi * idx / 1024.0
                re    = math.cos(angle)
                im    = math.sin(angle)

                fp8_re = self.float_to_fp8_e4m3(re) & 0xFF
                fp8_im = self.float_to_fp8_e4m3(im) & 0xFF
                fp4_re = self.float_to_fp4(re)       & 0x0F
                fp4_im = self.float_to_fp4(im)       & 0x0F

                word = (fp8_re << 16) | (fp8_im << 8) | (fp4_re << 4) | fp4_im
                f.write(f"{word:024b}\n")
        return path

    # ==================================================================
    # Test-vector file writer  (for reference; not used by this TB)
    # ==================================================================
    def _write_test_vectors_hex(self, sim_dir):
        path = os.path.join(sim_dir, 'test_vectors.hex')
        with open(path, 'w') as f:
            for vec in self.test_vectors:
                for sample in vec:
                    fp8_re = self.float_to_fp8_e4m3(sample.real) & 0xFF
                    fp8_im = self.float_to_fp8_e4m3(sample.imag) & 0xFF
                    f.write(f"{fp8_re:02x}{fp8_im:02x}\n")
        return path

    # ==================================================================
    # Testbench generator
    # ==================================================================
    def _generate_testbench(self, dut_file, design_name):
        design_name = self._sanitize_name(design_name)
        """
        Generate a Verilog-2001 compatible testbench that exercises the
        generated TOP module using the same interface as tb_fft_test.v:

            load_en / load_addr / load_data
            start / done
            unload_en / unload_addr / unload_data

        Outputs FP8 hex pairs (real, imag) one per line to sim/<dn>_output.txt.
        """
        n          = self.fft_size
        addr_bits  = int(math.log2(n))
        num_tests  = len(self.test_vectors)
        top_module = f"{design_name}_top"

        # Timing budgets
        butterflies     = (n // 2) * self.num_stages
        cycles_per_fft  = n + butterflies * 10 + n * 4 + 200
        ready_timeout   = 1024 + cycles_per_fft
        unload_cycles   = n * 5
        watchdog_ns     = (1024 + num_tests * (cycles_per_fft + n * 5) + 1000) * 10

        sim_dir  = os.path.abspath('./sim')
        out_path = os.path.join(sim_dir, f'{design_name}_output.txt').replace('\\', '/')

        # Build per-test input load lines
        vec_hex_lines = []
        for ti, vec in enumerate(self.test_vectors):
            for si, sample in enumerate(vec):
                fp8_re = self.float_to_fp8_e4m3(sample.real) & 0xFF
                fp8_im = self.float_to_fp8_e4m3(sample.imag) & 0xFF
                word   = (fp8_re << 8) | fp8_im
                vec_hex_lines.append(f"        tv[{ti*n + si}] = 16'h{word:04x};")
        vec_init = '\n'.join(vec_hex_lines)

        tb = f"""\
// Auto-generated testbench for {top_module}
// Mirrors tb_fft_test.v interface exactly.
`timescale 1ns/1ps

module tb_{design_name};

    reg        clk;
    reg        rst;
    reg        start;
    wire       done;

    reg        load_en;
    reg  [{addr_bits-1}:0]  load_addr;
    reg  [15:0] load_data;

    reg        unload_en;
    reg  [{addr_bits-1}:0]  unload_addr;
    wire [15:0] unload_data;

    integer i, ti, out_file;
    integer cycle_count;        // Cycles for the current FFT run
    integer total_cycles;       // Accumulated cycles across all test vectors
    integer load_cycles;        // Cycles for the current load phase
    integer unload_cycles_cnt;  // Cycles for the current unload phase

    // Test vector storage: fp8 packed {{real[7:0], imag[7:0]}}
    reg [15:0] tv [{num_tests*n - 1}:0];

    // DUT
    {top_module} dut (
        .clk        (clk),
        .rst        (rst),
        .start      (start),
        .done       (done),
        .load_en    (load_en),
        .load_addr  (load_addr),
        .load_data  (load_data),
        .unload_en  (unload_en),
        .unload_addr(unload_addr),
        .unload_data(unload_data)
    );

    // 100 MHz clock
    initial clk = 0;
    always  #5 clk = ~clk;

    // Watchdog
    initial begin
        #{watchdog_ns};
        $display("WATCHDOG TIMEOUT for {design_name}");
        $finish;
    end

    initial begin : STIM
        integer wait_cnt;

        // Pre-load test vectors
{vec_init}

        // Open output file
        out_file = $fopen("{out_path}", "w");

        // Initialise signals
        rst             = 0;
        start           = 0;
        load_en         = 0;
        load_addr       = 0;
        load_data       = 0;
        unload_en       = 0;
        unload_addr     = 0;
        total_cycles    = 0;

        // Hold reset for 8 cycles then release
        repeat(8) @(posedge clk);
        rst = 1;
        repeat(4) @(posedge clk);

        $display("\\n================================================================");
        $display("  Clock-Cycle Report  --  {design_name}  (FFT-{n})");
        $display("================================================================");
        $display("  %-4s  %-10s  %-12s  %-13s  %-12s",
                 "Test", "Load cyc", "Compute cyc", "Unload cyc", "Total cyc");
        $display("----------------------------------------------------------------");

        // Run each test vector
        for (ti = 0; ti < {num_tests}; ti = ti + 1) begin

            // --- Load phase ---
            @(posedge clk);
            load_cycles = 0;
            load_en = 1;
            for (i = 0; i < {n}; i = i + 1) begin
                load_addr = i[{addr_bits-1}:0];
                load_data = tv[ti*{n} + i];
                @(posedge clk);
                load_cycles = load_cycles + 1;
            end
            load_en = 0;

            @(posedge clk);
            load_cycles = load_cycles + 1;

            // --- Run FFT (count compute cycles: start pulse → done asserted) ---
            cycle_count = 0;
            start = 1;
            @(posedge clk);
            start = 0;
            cycle_count = cycle_count + 1;

            // Wait for done — count every clock edge
            wait_cnt = 0;
            while (!done && wait_cnt < {ready_timeout}) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
                wait_cnt    = wait_cnt    + 1;
            end
            if (!done)
                $display("WARN: done never asserted for test %0d, design {design_name}", ti);

            @(posedge clk);

            // --- Unload phase ---
            // Memory has 2-cycle read latency.
            // For each sample: assert address, wait 3 posedge clk, sample data.
            unload_cycles_cnt = 0;
            unload_en = 1;
            for (i = 0; i < {n}; i = i + 1) begin
                unload_addr = i[{addr_bits-1}:0];
                @(posedge clk); unload_cycles_cnt = unload_cycles_cnt + 1;
                @(posedge clk); unload_cycles_cnt = unload_cycles_cnt + 1;
                @(posedge clk); unload_cycles_cnt = unload_cycles_cnt + 1;
                $fwrite(out_file, "%04h\\n", unload_data);
            end
            unload_en = 0;

            @(posedge clk);
            @(posedge clk);

            // --- Per-test cycle report ---
            $display("  %-4d  %-10d  %-12d  %-13d  %-12d",
                     ti,
                     load_cycles,
                     cycle_count,
                     unload_cycles_cnt,
                     load_cycles + cycle_count + unload_cycles_cnt);
            total_cycles = total_cycles + load_cycles + cycle_count + unload_cycles_cnt;

        end // for ti

        $display("----------------------------------------------------------------");
        $display("  Total cycles (all %0d tests)  : %0d", {num_tests}, total_cycles);
        $display("  Avg   cycles per FFT compute  : %0d", total_cycles / {num_tests});
        $display("================================================================\\n");

        $fclose(out_file);
        $display("Simulation complete for {design_name}. Results in {out_path}");
        $finish;
    end

endmodule
"""
        tb_file = f'./sim/tb_{design_name}.v'
        os.makedirs('./sim', exist_ok=True)
        with open(tb_file, 'w') as f:
            f.write(tb)
        return tb_file

    # ==================================================================
    # Simulation runner
    # ==================================================================
    @staticmethod
    def _sanitize_name(name):
        """Replace characters invalid in Verilog identifiers."""
        import re
        return re.sub(r'[^A-Za-z0-9_]', '_', name)

    def run_verilog_simulation(self, verilog_file, design_name):
        design_name = self._sanitize_name(design_name)
        """
        Compile with iverilog and simulate with vvp.
        verilog_file  = path to the *core* .v file
        The matching *_top.v must sit in the same directory.
        Returns path to output file, or None on failure.
        """
        sim_dir = os.path.abspath('./sim')
        os.makedirs(sim_dir, exist_ok=True)

        self._write_twiddle_file(sim_dir)

        tb_file = self._generate_testbench(verilog_file, design_name)

        _exclude = {'fft_test.v', 'tb_fft_test.v'}
        lib_sources = sorted(
            f for f in glob_module.glob(
                os.path.join(self.verilog_sources_dir, '*.v')
            )
            if os.path.basename(f) not in _exclude
        )
        if not lib_sources:
            print(f"ERROR: No .v files found in {self.verilog_sources_dir}")
            return None

        top_file = verilog_file.replace('.v', '_top.v')
        extra    = [top_file] if os.path.exists(top_file) else []

        vvp_path = os.path.join(sim_dir, f'{design_name}.vvp')

        compile_cmd = (
            ['iverilog',
             '-o', vvp_path,
             '-I', os.path.abspath(self.verilog_sources_dir),
             '-g2012',
             tb_file,
             verilog_file]
            + extra
            + lib_sources
        )

        try:
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"iverilog FAILED for {design_name}:\n"
                      f"  stdout: {result.stdout[-3000:]}\n"
                      f"  stderr: {result.stderr[-3000:]}")
                return None

            sim_result = subprocess.run(
                ['vvp', vvp_path],
                capture_output=True, text=True,
                timeout=300,
                cwd=sim_dir
            )
            if sim_result.returncode != 0:
                print(f"vvp FAILED for {design_name}:\n"
                      f"  stdout: {sim_result.stdout[-3000:]}\n"
                      f"  stderr: {sim_result.stderr[-3000:]}")
                return None

            for line in sim_result.stdout.splitlines():
                if any(kw in line for kw in ('ERROR', 'WARN', 'WATCHDOG', 'TIMEOUT')):
                    print(f"SIM [{design_name}]: {line}")

            return os.path.join(sim_dir, f'{design_name}_output.txt')

        except FileNotFoundError as e:
            print(f"Simulator not found: {e}\n  Ensure iverilog/vvp are on PATH.")
            return None
        except subprocess.TimeoutExpired:
            print(f"Simulation timeout for {design_name}")
            return None
        except Exception as e:
            print(f"Simulation error for {design_name}: {e}")
            return None

    # ==================================================================
    # Output parser
    # ==================================================================
    def _parse_simulation_output(self, output_file, final_stage_is_fp8=True):
        """
        Parse 4-hex-digit lines written by the testbench.

        If final stage is FP8:
            Each line is {fp8_real[7:0], fp8_imag[7:0]} — upper 8 bits real,
            lower 8 bits imag.

        If final stage is FP4:
            Each line is {8'h00, fp4_real[3:0], fp4_imag[3:0]} — bits [7:4]
            real, bits [3:0] imag.
        """
        outputs = []
        try:
            with open(output_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    word = int(line, 16) & 0xFFFF
                    if final_stage_is_fp8:
                        fp8_real = (word >> 8) & 0xFF
                        fp8_imag =  word       & 0xFF
                        real = self.fp8_to_float(fp8_real)
                        imag = self.fp8_to_float(fp8_imag)
                    else:
                        fp4_real = (word >> 4) & 0xF
                        fp4_imag =  word       & 0xF
                        real = self.fp4_to_float(fp4_real)
                        imag = self.fp4_to_float(fp4_imag)
                    outputs.append(real + 1j * imag)
        except FileNotFoundError:
            print(f"Simulation output file not found: {output_file}")
            return None
        except Exception as e:
            print(f"Error parsing {output_file}: {e}")
            return None
        return np.array(outputs) if outputs else None

    # ==================================================================
    # Metrics
    # ==================================================================
    def calculate_sqnr(self, golden, approximate):
        """
        Signal-to-Quantisation-Noise Ratio (dB).

            SQNR = 10 * log10( signal_power / noise_power )

        where
            signal_power = mean( |golden|^2 )
            noise_power  = mean( |golden - approximate|^2 )

        SQNR is the standard metric for fixed/low-precision arithmetic:
          - Normalises noise relative to signal energy, not peak value.
          - An all-FP8 design typically yields SQNR ~30-40 dB.
          - An all-FP4 design typically yields SQNR ~15-25 dB.
          - Returns +inf when noise is zero (exact result).
          - Returns 0 dB when signal power equals noise power.
          - Can be negative (dB) when noise exceeds signal.
        """
        noise_power = np.mean(np.abs(golden - approximate) ** 2)
        if noise_power == 0:
            return float('inf')
        signal_power = np.mean(np.abs(golden) ** 2)
        if signal_power == 0:
            return 0.0
        return 10.0 * np.log10(signal_power / noise_power)

    # ==================================================================
    # Top-level entry point
    # ==================================================================
    def evaluate_design(self, verilog_file, design_name, chromosome=None):
        """
        Run simulation and return avg_sqnr_dB (single float).
        On failure returns -100.0.

        Prints a per-signal SQNR breakdown table so each signal's
        contribution to the average is visible in the optimisation log.

        chromosome: the precision chromosome for this design.
                    Last gene (chromosome[-1]) determines final stage
                    adder precision: 1=FP8, 0=FP4.
                    If None, defaults to FP8 output parsing.
        """
        design_name = self._sanitize_name(design_name)

        # Determine final stage output precision from chromosome
        # Chromosome format: [s0_mult, s0_add, s1_mult, s1_add, ..., sN_mult, sN_add]
        # Last gene is final stage adder precision: 1=FP8, 0=FP4
        final_stage_is_fp8 = True
        if chromosome is not None:
            final_stage_is_fp8 = bool(chromosome[-1])

        output_file = self.run_verilog_simulation(verilog_file, design_name)
        if output_file is None:
            return -100.0

        sim_outputs = self._parse_simulation_output(output_file, final_stage_is_fp8)
        if sim_outputs is None or len(sim_outputs) == 0:
            return -100.0

        n         = self.fft_size
        num_tests = len(self.test_vectors)

        total_sqnr = 0.0
        valid      = 0

        print(f"\n{'─'*55}")
        print(f"  SQNR breakdown  —  {design_name}  (FFT-{n})")
        print(f"{'─'*55}")
        print(f"  {'Signal':<22}  {'SQNR (dB)':>10}")
        print(f"{'─'*55}")

        for i in range(min(num_tests, len(sim_outputs) // n)):
            approx = sim_outputs[i * n : (i + 1) * n]
            golden = self.golden_outputs[i]
            label  = SIGNAL_LABELS[i] if i < len(SIGNAL_LABELS) else f"Signal {i}"

            sqnr = self.calculate_sqnr(golden, approx)

            if math.isinf(sqnr):
                sqnr_str = "  inf  (exact)"
                # Treat as very high but finite for averaging
                total_sqnr += 100.0
                valid += 1
            elif math.isnan(sqnr):
                sqnr_str = "         NaN"
                # Skip NaN from average
            else:
                sqnr_str = f"{sqnr:>10.2f} dB"
                total_sqnr += sqnr
                valid += 1

            print(f"  {label:<22}  {sqnr_str}")

        print(f"{'─'*55}")

        if valid == 0:
            print(f"  {'Average SQNR':<22}  {'N/A':>10}")
            print(f"{'─'*55}\n")
            return -100.0

        avg_sqnr = total_sqnr / valid
        print(f"  {'Average SQNR':<22}  {avg_sqnr:>10.2f} dB")
        print(f"{'─'*55}\n")

        return avg_sqnr


# =============================================================================
# Quick smoke-test
# =============================================================================
def test_evaluator():
    ev = PerformanceEvaluator(fft_size=8)
    print(f"Test vectors : {len(ev.test_vectors)}")
    print(f"Signal labels: {SIGNAL_LABELS}")
    print(f"FFT size     : {ev.fft_size}")
    print(f"FP4 0b0101   -> {ev.fp4_to_float(0b0101):.4f}  (expect 1.5)")
    print(f"FP8 0b01000011 -> {ev.fp8_to_float(0b01000011):.6f}  (expect ~1.375)")
    print(f"FP8(1.0) encode -> 0x{ev.float_to_fp8_e4m3(1.0):02x}  (expect 0x38)")
    print(f"FP8(0.0) encode -> 0x{ev.float_to_fp8_e4m3(0.0):02x}  (expect 0x00)")

    # Verify SQNR with a perfect (zero-noise) pair
    golden = np.array([1+1j, -1+0j, 0-1j, 1+1j], dtype=complex)
    sqnr_perfect = ev.calculate_sqnr(golden, golden)
    print(f"\nSQNR (perfect match) -> {sqnr_perfect}  (expect inf)")

    # Verify SQNR with half-amplitude approximation (~6 dB)
    sqnr_half = ev.calculate_sqnr(golden, golden * 0.5)
    print(f"SQNR (half amplitude) -> {sqnr_half:.2f} dB  (expect ~-6 dB for noise=0.5*signal)")

    print("\nGolden FFT outputs (FP8-quantised inputs):")
    for i, (label, g) in enumerate(zip(SIGNAL_LABELS, ev.golden_outputs)):
        sig_power = np.mean(np.abs(g) ** 2)
        print(f"  [{i}] {label:<22}  signal_power={sig_power:.4f}")


if __name__ == "__main__":
    test_evaluator()
