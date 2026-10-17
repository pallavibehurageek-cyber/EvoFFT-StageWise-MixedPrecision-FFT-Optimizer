# EvoFFT: Evolutionary Multi-Objective Design Space Exploration for Mixed-Precision FFT Architectures

This repository contains the complete framework for EvoFFT, an automated design pipeline that utilizes NSGA-II (Non-dominated Sorting Genetic Algorithm II) to explore the trade-offs between computational fidelity and hardware efficiency in Fast Fourier Transform (FFT) processors. By transitioning from uniform-precision architectures to mixed-precision heterogeneity, this framework generates signal-processing cores that are sufficiently accurate for power-constrained edge environments.

---

## 1. Research Overview

As the demand for edge processing increases, the need for on-chip digital signal processing (DSP) rises. Traditional FFT designs rely on uniform-precision arithmetic (e.g., IEEE-754), which often results in systematic over-engineering and unsustainable area and power costs for constrained deployments.

EvoFFT exploits the stage-dependent error sensitivity of radix-2 FFT butterfly structures. Errors introduced in earlier stages propagate more significantly than those in later stages. Specifically, an error at stage \( S \) propagates to \( 2^{\log_2 N - S} \) downstream bins. This asymmetry enables selective allocation of higher precision (FP8) in critical stages and lower precision (FP4) elsewhere.

---

## 2. Primary Contributions

- **Hardware-Grounded Encoding**  
  Per-stage mixed-precision mapping ensuring each design corresponds to a synthesizable FPGA implementation.

- **Multi-Objective Optimization**  
  Simultaneous minimization of dynamic power, LUT area, SQNR degradation, and latency.

- **Post-Implementation Validation**  
  Integration with Xilinx Vivado for accurate power and area estimation.

- **Scalable FFT Library**  
  Optimized FFT cores from \( N = 8 \) to \( N = 1024 \), achieving up to 79\% logic reduction compared to conventional designs.

---

## 3. System Architecture

The framework uses a parameterized hardware template supporting per-stage precision scaling between FP4 (E2M1) and FP8 (E4M3).

### 3.1 Hardware Core Features

- **Mixed-Precision Butterfly Units**  
  Four variants covering all FP4/FP8 combinations, selected via Verilog generate constructs.

- **Unified 24-bit Memory Architecture**  
  Dual-bank ping-pong BRAM storing both formats simultaneously, eliminating conversion overhead.

- **Address Generation Unit (AGU)**  
  Supports variable FFT sizes up to \( N = 1024 \) with dynamic indexing.

- **Twiddle Factor ROM**  
  \( 512 \times 24 \)-bit ROM with scalable addressing for different FFT sizes.

---

## 4. Optimization Pipeline

The pipeline treats stage precision as a structured, high-dimensional design variable. A Python-based template generator is combined with evolutionary search across a design space of size:

\[
4^{\log_2(N)}
\]

### 4.1 NSGA-II Strategy

- **Chromosome Encoding**

\[
C = [m_0, a_0, m_1, a_1, \dots, m_{S-1}, a_{S-1}]
\]

where:
- \( m_i \): multiplier precision at stage \( i \)  
- \( a_i \): adder precision at stage \( i \)

- **Performance Evaluation**

Automated SQNR analysis is performed using diverse spectral inputs, including:
- Impulse
- Single tone
- Multi-tone
- Chirp (LFM)
- Gaussian pulse
- Radar Barker-13
- Doppler burst

- **Objective Weighting**

\[
Power = 1.0,\quad Area = 0.001,\quad SQNR = 50.0,\quad Latency = 8.0
\]

---

## 5. Performance and Results

Experiments on the ZedBoard (Zynq-7000 SoC) show that optimal configurations frequently retain FP8 multipliers in intermediate stages while using FP4 adders, indicating that multiplier quantization dominates noise behavior.

### 5.1 Benchmark Summary (\( N = 1024 \))

| Metric    | EvoFFT (Balanced) | Heo et al. (HFP 16-bit) | Sahu et al. (11-bit Float) |
|----------|------------------|--------------------------|----------------------------|
| LUT Area | 2,280            | 10,891                   | 4,215                      |
| Power    | ~0.106 W         | Variable                 | Variable                   |
| SQNR     | 29.62 dB         | -                        | -                          |

A key property of the architecture is the sequential butterfly reuse strategy, which maintains an approximately constant power profile (~0.106 W) across FFT sizes.

---

## 6. Project Structure

```
verilog_sources/        # Core RTL modules (adders, multipliers, AGU)
generated_designs/      # Auto-generated FFT architectures
sim/                    # Simulation environment and testbenches

fft_template_generator.py
performance_evaluator.py
runMixedFFTOptimization.py
vivado_synthesis.tcl
```

---

## 7. Results and Applications

The generated FFT cores are suitable for:

- Edge computing systems
- Biomedical signal processing
- Radar signal analysis
- Energy-constrained embedded platforms

---
