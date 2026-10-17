"""
Mixed-Precision FFT Template Generator
=======================================
Generates per-solution core + top Verilog files for a DIT radix-2 FFT.

Design includes:
  - Active-low async reset (negedge rst)
  - Mixed-precision butterfly_wrapper (parameterised per stage)
  - Unified 24-bit ping-pong memory  (mixed_memory_unified)
  - Single AGU (dit_fft_agu_variable)
  - Bit-reversal on load  (bit_reverse)
  - Single twiddle ROM    (twiddle_factor_unified)
  - 2-cycle memory read latency → READ_A/READ_B/WAIT_A/WAIT_B

Top module wraps the core and exposes a streaming 24-bit I/O interface
compatible with the performance evaluator testbench.

Chromosome encoding (2 genes per stage):
  gene[2*s]   → MULT_PRECISION for stage s  (0=FP4, 1=FP8)
  gene[2*s+1] → ADD_PRECISION  for stage s  (0=FP4, 1=FP8)
"""

import os
import math


class FFTTemplateGenerator:
    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self, fft_size):
        self.fft_size          = fft_size
        self.num_stages        = int(math.log2(fft_size))
        self.addr_width        = 11          # fixed 10-bit address (matches AGU/memory)
        self.butterflies_per_stage = fft_size // 2
        self.total_butterflies = self.butterflies_per_stage * self.num_stages
        self.chromosome_length = self.num_stages * 2
        self.MAX_N_HW          = 1024

        print(f"FFTTemplateGenerator FFT-{fft_size}:")
        print(f"  Stages            : {self.num_stages}")
        print(f"  Butterflies/stage : {self.butterflies_per_stage}")
        print(f"  Chromosome length : {self.chromosome_length}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_chromosome_length(self):
        return self.chromosome_length

    def chromosome_to_config(self, chromosome):
        """Convert flat chromosome list to structured stage configuration."""
        config = {
            'fft_size'  : self.fft_size,
            'num_stages': self.num_stages,
            'addr_width': self.addr_width,
            'stages'    : [],
            'MAX_N_HW'  : self.MAX_N_HW,
        }
        for stage in range(self.num_stages):
            idx       = stage * 2
            mult_prec = int(chromosome[idx])     if idx     < len(chromosome) else 0
            add_prec  = int(chromosome[idx + 1]) if idx + 1 < len(chromosome) else 0
            out_prec  = add_prec   

            config['stages'].append({
                'stage_num'       : stage,
                'mult_precision'  : mult_prec,
                'add_precision'   : add_prec,
                'output_precision': out_prec,
            })
        return config

    # ------------------------------------------------------------------
    # File generation helpers
    # ------------------------------------------------------------------~
    def generate_verilog(self, chromosome, output_file):
        """
        Generate core + top into the directory of output_file.
        Returns (core_file_path, top_file_path).
        """
        config = self.chromosome_to_config(chromosome)

        out_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(out_dir, exist_ok=True)

        stem             = os.path.splitext(os.path.basename(output_file))[0]
        core_module_name = f"{stem}_core"
        top_module_name  = f"{stem}_top"

        core_code = self._generate_core(config, core_module_name)
        with open(output_file, 'w') as f:
            f.write(core_code)

        top_file = os.path.join(out_dir, f"{stem}_top.v")
        top_code = self._generate_top(config, core_module_name, top_module_name)
        with open(top_file, 'w') as f:
            f.write(top_code)

        return output_file, top_file

    def generate_complete_fft(self, chromosome, output_dir='./generated_designs'):
        """Convenience wrapper; returns top file path."""
        config = self.chromosome_to_config(chromosome)
        os.makedirs(output_dir, exist_ok=True)

        base             = f"mixed_fft_{self.fft_size}"
        core_module_name = f"{base}_core"
        top_module_name  = f"{base}_top"

        core_file = f"{output_dir}/{base}_core.v"
        top_file  = f"{output_dir}/{base}_top.v"

        with open(core_file, 'w') as f:
            f.write(self._generate_core(config, core_module_name))
        with open(top_file, 'w') as f:
            f.write(self._generate_top(config, core_module_name, top_module_name))

        print(f"✓ Generated: {core_file}")
        print(f"✓ Generated: {top_file}")
        return top_file

    def analyze_chromosome_statistics(self, chromosome):
        config   = self.chromosome_to_config(chromosome)
        fp8_mult = sum(s['mult_precision'] for s in config['stages'])
        fp8_add  = sum(s['add_precision']  for s in config['stages'])
        return {
            'fp8_mult'   : fp8_mult,
            'fp4_mult'   : self.num_stages - fp8_mult,
            'fp8_add'    : fp8_add,
            'fp4_add'    : self.num_stages - fp8_add,
            'stage_stats': [
                {
                    'stage'   : s['stage_num'],
                    'fp8_mult': s['mult_precision'],
                    'fp4_mult': 1 - s['mult_precision'],
                    'fp8_add' : s['add_precision'],
                    'fp4_add' : 1 - s['add_precision'],
                }
                for s in config['stages']
            ],
        }

    # ==================================================================
    # Core Verilog generator
    # ==================================================================
    def _generate_core(self, config, core_module_name):
        n    = config['fft_size']
        aw   = self.addr_width        # always 10
        ns   = config['num_stages']
        MAXn = config['MAX_N_HW']
        stages = config['stages']

        # ---- stage localparams ----
        lparams_lines = []
        for s in stages:
            sn = s['stage_num']
            lparams_lines += [
                f"    localparam STAGE{sn}_MULT_PREC = {s['mult_precision']};",
                f"    localparam STAGE{sn}_ADD_PREC  = {s['add_precision']};",
                f"    localparam STAGE{sn}_OUT_PREC  = {s['output_precision']};",
            ]
        lparams = '\n'.join(lparams_lines)

        # ---- output precision of last stage (for read-back) ----
        last_out_prec = stages[-1]['output_precision']

        # ---- precision mux (combinational, driven by curr_stage) ----
        prec_cases = []
        for s in stages:
            sn       = s['stage_num']
            # Read precision = output precision of previous stage
            # (stage 0 reads the loaded data which is always FP8)
            rd_prec  = "1'b1" if sn == 0 else f"STAGE{sn-1}_OUT_PREC"
            rd_comment = "// Input always loaded as FP8" if sn == 0 else f"// previous stage output precision"
            prec_cases.append(
                f"            10'd{sn}: begin\n"
                f"                cur_mult_prec = STAGE{sn}_MULT_PREC;\n"
                f"                cur_add_prec  = STAGE{sn}_ADD_PREC;\n"
                f"                cur_rd_prec   = {rd_prec};  {rd_comment}\n"
                f"                cur_wr_prec   = STAGE{sn}_OUT_PREC;\n"
                f"            end"
            )

        prec_mux = (
            "    always @(*) begin\n"
            "        if (ext_reading) begin\n"
            f"            cur_mult_prec = 1'b0;\n"
            f"            cur_add_prec  = 1'b0;\n"
            f"            cur_rd_prec   = 1'b{last_out_prec};  // Read FFT result at final-stage precision ({last_out_prec}=FP{'8' if last_out_prec else '4'})\n"
            f"            cur_wr_prec   = 1'b0;\n"
            "        end else begin\n"
            "            case (curr_stage)\n"
            + '\n'.join(prec_cases) + "\n"
            "                default: begin\n"
            "                    cur_mult_prec = 1'b0; cur_add_prec = 1'b0;\n"
            "                    cur_rd_prec = 1'b1; cur_wr_prec = 1'b0;\n"
            "                end\n"
            "            endcase\n"
            "        end\n"
            "    end"
        )

        pad_bits = aw - int(math.log2(n))   # may be 0 for n==MAX_N

        # ---- per-stage butterfly instantiations ----
        bf_lines = ["    // Per-stage butterfly wrappers (precision baked in)"]
        for s in stages:
            sn = s['stage_num']
            bf_lines += [
                f"    wire [15:0] X_st{sn}, Y_st{sn};",
                f"    wire        fp8_out_st{sn};",
            ]
        bf_lines.append("")
        for s in stages:
            sn = s['stage_num']
            mp = s['mult_precision']
            ap = s['add_precision']
            bf_lines += [
                f"    butterfly_wrapper #(",
                f"        .MULT_PRECISION({mp}),",
                f"        .ADD_PRECISION ({ap})",
                f"    ) bf_st{sn} (",
                f"        .A            (A_24),",
                f"        .B            (B_24),",
                f"        .W            (twiddle),",
                f"        .X            (X_st{sn}),",
                f"        .Y            (Y_st{sn}),",
                f"        .output_is_fp8(fp8_out_st{sn})",
                f"    );",
                "",
            ]
        # Mux: select active butterfly output
        bf_lines += [
            "    // Select butterfly output for the active stage",
            "    reg [15:0] X_bf, Y_bf;",
            "    reg        bf_is_fp8;",
            "    always @(*) begin",
            "        case (curr_stage)",
        ]
        for s in stages:
            sn = s['stage_num']
            bf_lines += [
                f"            10'd{sn}: begin X_bf = X_st{sn}; Y_bf = Y_st{sn}; bf_is_fp8 = fp8_out_st{sn}; end",
            ]
        bf_lines += [
            "            default: begin X_bf = 16'h0; Y_bf = 16'h0; bf_is_fp8 = 1'b0; end",
            "        endcase",
            "    end",
        ]
        butterfly_block = '\n'.join(bf_lines)

        # ---- 16→24 expansion for butterfly inputs ----
        # Memory outputs 16-bit (FP8 or FP4 zero-padded).
        # butterfly_wrapper expects 24-bit unified format:
        #   [23:16]=FP8_real, [15:8]=FP8_imag, [7:4]=FP4_real, [3:0]=FP4_imag
        mem_expand = (
            "    // Expand 16-bit memory read to full 24-bit unified format\n"
            "    // [23:8] = FP8 complex, [7:0] = FP4 complex\n"
            "    // Both slots must always be populated so any butterfly_wrapper\n"
            "    // configuration can read the correct slice.\n"
            "\n"
            "    // Cross-convert: FP8 read → downconvert to fill FP4 slot\n"
            "    wire [7:0]  rd_fp8_as_fp4;\n"
            "    complex_fp8_to_fp4 rd_conv_down (\n"
            "        .complex_fp8(rd_data_16),\n"
            "        .complex_fp4(rd_fp8_as_fp4)\n"
            "    );\n"
            "\n"
            "    // Cross-convert: FP4 read → upconvert to fill FP8 slot\n"
            "    wire [15:0] rd_fp4_as_fp8;\n"
            "    complex_fp4_to_fp8 rd_conv_up (\n"
            "        .complex_fp4(rd_data_16[7:0]),\n"
            "        .complex_fp8(rd_fp4_as_fp8)\n"
            "    );\n"
            "\n"
            "    wire [23:0] mem_rd_24;\n"
            "    assign mem_rd_24 = cur_rd_prec\n"
            "                       ? {rd_data_16,    rd_fp8_as_fp4}   // FP8 primary + FP4 downconverted\n"
            "                       : {rd_fp4_as_fp8, rd_data_16[7:0]};// FP8 upconverted + FP4 primary\n"
            "\n"
            "    reg [23:0] A_24, B_24;"
        )

        # ---- write-back packing ----
        # We always write 24-bit unified format.
        # If butterfly output is FP8 (bf_is_fp8): result is in X_bf[15:0]
        #   → pack as {X_bf[15:0], convert_to_fp4(X_bf)}
        # If FP4: result is in X_bf[7:0]
        #   → pack as {convert_to_fp8(X_bf[7:0]), X_bf[7:0]}
        writeback_block = (
            "    // Write-back: always store both FP8 and FP4 slots\n"
            "    wire [7:0]  X_fp4_packed, Y_fp4_packed;\n"
            "    wire [15:0] X_fp8_packed, Y_fp8_packed;\n"
            "\n"
            "    // FP8→FP4 converters (used when butterfly output is FP8)\n"
            "    fp8_to_fp4_converter conv_xr (.fp8_in(X_reg[15:8]), .fp4_out(X_fp4_packed[7:4]));\n"
            "    fp8_to_fp4_converter conv_xi (.fp8_in(X_reg[7:0]),  .fp4_out(X_fp4_packed[3:0]));\n"
            "    fp8_to_fp4_converter conv_yr (.fp8_in(Y_reg[15:8]), .fp4_out(Y_fp4_packed[7:4]));\n"
            "    fp8_to_fp4_converter conv_yi (.fp8_in(Y_reg[7:0]),  .fp4_out(Y_fp4_packed[3:0]));\n"
            "\n"
            "    // FP4→FP8 converters (used when butterfly output is FP4)\n"
            "    fp4_to_fp8_converter conv_xr8 (.fp4_in(X_reg[7:4]), .fp8_out(X_fp8_packed[15:8]));\n"
            "    fp4_to_fp8_converter conv_xi8 (.fp4_in(X_reg[3:0]), .fp8_out(X_fp8_packed[7:0]));\n"
            "    fp4_to_fp8_converter conv_yr8 (.fp4_in(Y_reg[7:4]), .fp8_out(Y_fp8_packed[15:8]));\n"
            "    fp4_to_fp8_converter conv_yi8 (.fp4_in(Y_reg[3:0]), .fp8_out(Y_fp8_packed[7:0]));\n"
            "\n"
            "    // Final 24-bit write data\n"
            "    wire [23:0] X_wr_24, Y_wr_24;\n"
            "    assign X_wr_24 = out_was_fp8 ? {X_reg,       X_fp4_packed}\n"
            "                                 : {X_fp8_packed, X_reg[7:0]};\n"
            "    assign Y_wr_24 = out_was_fp8 ? {Y_reg,       Y_fp4_packed}\n"
            "                                 : {Y_fp8_packed, Y_reg[7:0]};"
        )

        return f"""\
// =============================================================================
// Mixed-Precision FFT Core – {n}-point
// Auto-generated by FFTTemplateGenerator
// Active-low asynchronous reset (negedge rst).
// =============================================================================
`timescale 1ns/1ps

module {core_module_name} #(
    parameter MAX_N      = {MAXn},
    parameter ADDR_WIDTH = {aw}
)(
    input  wire        clk,
    input  wire        rst,     // active-low async reset

    // Control
    input  wire        start,
    output reg         done,

    // External load interface (write input samples, bit-reversed externally)
    input  wire                  ext_wr_en,
    input  wire [ADDR_WIDTH-1:0] ext_wr_addr,
    input  wire [23:0]           ext_wr_data,  // 24-bit unified format

    // External unload interface
    input  wire                  ext_reading,
    input  wire [ADDR_WIDTH-1:0] ext_rd_addr,
    output wire [15:0]           ext_rd_data,  // 16-bit (precision = last stage output)

    // bank_sel driven by top during load/unload
    input  wire                  ext_bank_sel
);

    // =========================================================================
    // FSM states
    // =========================================================================
    localparam IDLE           = 4'd0,
               START_AGU      = 4'd1,
               WAIT_AGU_START = 4'd2,
               READ_A         = 4'd3,
               READ_B         = 4'd4,
               WAIT_A         = 4'd5,
               WAIT_B         = 4'd6,
               COMPUTE        = 4'd7,
               WRITE_X        = 4'd8,
               WRITE_Y        = 4'd9,
               WAIT_AGU       = 4'd10,
               EVAL_AGU       = 4'd11,
               DONE_STATE     = 4'd12;

    reg [3:0] state;

    // =========================================================================
    // Per-stage precision localparams (baked in from chromosome)
    // =========================================================================
{lparams}

    // =========================================================================
    // Runtime precision signals (combinational, from curr_stage)
    // =========================================================================
    reg cur_mult_prec;
    reg cur_add_prec;
    reg cur_rd_prec;
    reg cur_wr_prec;

    // =========================================================================
    // AGU
    // =========================================================================
    reg  start_agu_reg;
    reg  next_step_reg;
    wire [ADDR_WIDTH-1:0] idx_a, idx_b, k;
    wire done_stage, done_fft;
    wire [ADDR_WIDTH-1:0] curr_stage;

    dit_fft_agu_variable #(
        .MAX_N     ({MAXn}),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) agu (
        .clk          (clk),
        .reset        (rst),
        .start        (start_agu_reg),
        .N            ({aw}'d{n}),
        .next_step    (next_step_reg),
        .idx_a        (idx_a),
        .idx_b        (idx_b),
        .k            (k),
        .done_stage   (done_stage),
        .done_fft     (done_fft),
        .curr_stage   (curr_stage),
        .twiddle_output()
    );

    // =========================================================================
    // Precision mux
    // =========================================================================
{prec_mux}

    // =========================================================================
    // Twiddle ROM  – precision follows cur_mult_prec
    // =========================================================================
    wire [15:0] twiddle;

    twiddle_factor_unified #(
        .MAX_N     ({MAXn}),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) twiddle_rom (
        .k          (k),
        .n          ({aw}'d{n}),
        .PRECISION  (cur_mult_prec),
        .twiddle_out(twiddle)
    );

    // =========================================================================
    // Memory signals
    // =========================================================================
    // bank_sel: during FFT computation the core owns it;
    //           during IDLE (load) and external read the top drives ext_bank_sel.
    reg  fft_bank_sel;
    wire active_bank_sel = (state == IDLE || ext_reading) ? ext_bank_sel : fft_bank_sel;

    // Read address mux: external read takes priority
    wire [ADDR_WIDTH-1:0] mem_rd_addr = ext_reading ? ext_rd_addr :
                                        (state == READ_A) ? idx_a :
                                        (state == READ_B) ? idx_b : {aw}'d0;

    // Write mux: external load takes priority; otherwise butterfly results
    wire        mem_wr_en   = ext_wr_en ? 1'b1 :
                              (state == WRITE_X || state == WRITE_Y) ? 1'b1 : 1'b0;

    wire [ADDR_WIDTH-1:0] mem_wr_addr = ext_wr_en ? ext_wr_addr :
                                        (state == WRITE_X) ? idx_a :
                                        (state == WRITE_Y) ? idx_b : {aw}'d0;

    wire [23:0] mem_wr_data = ext_wr_en ? ext_wr_data :
                              (state == WRITE_X) ? X_wr_24 :
                              (state == WRITE_Y) ? Y_wr_24 : 24'd0;

    wire [15:0] rd_data_16;

    mixed_memory_unified #(
        .n         ({MAXn}),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) mem (
        .clk          (clk),
        .rst          (rst),
        .bank_sel     (active_bank_sel),
        .rd_addr_0    (mem_rd_addr),
        .rd_precision_0(cur_rd_prec),
        .rd_data_0    (rd_data_16),
        .wr_en_1      (mem_wr_en),
        .wr_addr_1    (mem_wr_addr),
        .wr_data_1    (mem_wr_data)
    );

    assign ext_rd_data = rd_data_16;

    // =========================================================================
    // Expand 16-bit memory read to 24-bit butterfly inputs
    // =========================================================================
{mem_expand}

    // =========================================================================
    // Twiddle ROM already instantiated above
    // =========================================================================

    // =========================================================================
    // Per-stage butterfly instances
    // =========================================================================
{butterfly_block}

    // =========================================================================
    // Write-back packing (FP4↔FP8 converters)
    // =========================================================================
    reg [15:0] X_reg, Y_reg;
    reg        out_was_fp8;

{writeback_block}

    // =========================================================================
    // FSM
    // =========================================================================
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            state         <= IDLE;
            start_agu_reg <= 1'b0;
            next_step_reg <= 1'b0;
            fft_bank_sel  <= 1'b0;
            done          <= 1'b0;
            A_24          <= 24'd0;
            B_24          <= 24'd0;
            X_reg         <= 16'd0;
            Y_reg         <= 16'd0;
            out_was_fp8   <= 1'b0;
        end else begin
            case (state)
                // -----------------------------------------------------------
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        fft_bank_sel <= 1'b0; // Stage 0 reads Bank 0, writes Bank 1
                        state        <= START_AGU;
                    end
                end

                // -----------------------------------------------------------
                START_AGU: begin
                    start_agu_reg <= 1'b1;
                    state         <= WAIT_AGU_START;
                end

                WAIT_AGU_START: begin
                    start_agu_reg <= 1'b0;
                    state         <= READ_A;
                end

                // -----------------------------------------------------------
                // 2-cycle synchronous memory read sequence
                // Cycle 1 (READ_A): address idx_a is presented to memory
                // Cycle 2 (READ_B): address idx_b is presented; A data available next cycle
                // Cycle 3 (WAIT_A): A data sampled from rd_data_16
                // Cycle 4 (WAIT_B): B data sampled from rd_data_16
                // -----------------------------------------------------------
                READ_A: begin
                    // mem_rd_addr is idx_a (driven combinationally)
                    state <= READ_B;
                end

                READ_B: begin
                    // mem_rd_addr is idx_b
                    state <= WAIT_A;
                end

                WAIT_A: begin
                    // A data is now stable on rd_data_16
                    A_24  <= mem_rd_24;
                    state <= WAIT_B;
                end

                WAIT_B: begin
                    // B data is now stable on rd_data_16
                    B_24        <= mem_rd_24;
                    state       <= COMPUTE;
                end

                COMPUTE: begin
                    // Capture butterfly outputs (combinational from A_24/B_24)
                    X_reg       <= X_bf;
                    Y_reg       <= Y_bf;
                    out_was_fp8 <= bf_is_fp8;
                    state       <= WRITE_X;
                end

                // -----------------------------------------------------------
                WRITE_X: begin
                    // mem_wr_en=1, write X_wr_24 to idx_a  (combinational)
                    state <= WRITE_Y;
                end

                WRITE_Y: begin
                    // mem_wr_en=1, write Y_wr_24 to idx_b
                    next_step_reg <= 1'b1;
                    state         <= WAIT_AGU;
                end

                WAIT_AGU: begin
                    next_step_reg <= 1'b0;
                    state         <= EVAL_AGU;
                end

                EVAL_AGU: begin
                    if (done_fft) begin
                        // After {ns} stages (each stage flips bank), result is in bank {ns % 2}
                        fft_bank_sel <= 1'b{ns % 2};
                        state        <= DONE_STATE;
                    end else if (done_stage) begin
                        fft_bank_sel <= ~fft_bank_sel; // Ping-pong
                        state        <= READ_A;
                    end else begin
                        state        <= READ_A;
                    end
                end

                // -----------------------------------------------------------
                DONE_STATE: begin
                    done <= 1'b1;
                    if (!start) state <= IDLE;
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
"""

    # ==================================================================
    # Top Verilog generator
    # ==================================================================
    def _generate_top(self, config, core_module_name, top_module_name):
        n    = config['fft_size']
        aw   = self.addr_width
        ns   = config['num_stages']
        MAXn = config['MAX_N_HW']
        # addr_bits for load/unload address ports
        addr_bits = int(math.log2(n))
        pad_bits  = aw - addr_bits          # 0 when n == MAX_N
        last_out_prec = config['stages'][-1]['output_precision']
        # When pad_bits == 0 (N==MAX_N), load_addr is already aw-bits wide
        if pad_bits > 0:
            br_in_expr = "{{ " + str(pad_bits) + "'b0, load_addr }}"
            unload_addr_expr = "{{ " + str(pad_bits) + "'b0, unload_addr }}"
        else:
            br_in_expr = "load_addr"
            unload_addr_expr = "unload_addr"
        addr_msb = addr_bits - 1  # for port declarations [addr_msb:0]

        return f"""\
// =============================================================================
// Mixed-Precision FFT TOP – {n}-point
// Instantiates: {core_module_name}
// Auto-generated by FFTTemplateGenerator
//
// Interface:
//   load_en / load_addr / load_data   → write input samples
//   start / done                      → FFT trigger / completion
//   unload_en / unload_addr / unload_data → read FFT results
// =============================================================================
`timescale 1ns/1ps

module {top_module_name} (
    input  wire        clk,
    input  wire        rst,     // active-low async reset

    // Control
    input  wire        start,
    output reg         done,

    // Load interface  (load_data is 16-bit FP8: {{real[7:0], imag[7:0]}})
    input  wire              load_en,
    input  wire [{addr_msb}:0]  load_addr,
    input  wire [15:0]       load_data,

    // Unload interface (2-cycle read latency)
    input  wire              unload_en,
    input  wire [{addr_msb}:0]  unload_addr,
    output wire [15:0]       unload_data
);

    // -------------------------------------------------------------------------
    // Bit-reversal for load addressing
    // -------------------------------------------------------------------------
    wire [{aw-1}:0] load_addr_rev;

    bit_reverse #(
        .MAX_N({MAXn}),
        .WIDTH({aw})
    ) br (
        .in  ({br_in_expr}),
        .N   ({aw}'d{n}),
        .out (load_addr_rev)
    );

    // -------------------------------------------------------------------------
    // Bank-select management
    //   IDLE / loading : bank_sel = 1'b1 (so writes go to Bank 0)
    //   During FFT     : core manages internally
    //   After FFT      : bank_sel = {ns}[0] ? 1'b1 : 1'b0
    //                    (result ends up in bank = num_stages MOD 2)
    // -------------------------------------------------------------------------
    reg bank_sel;

    // -------------------------------------------------------------------------
    // Core instantiation
    // -------------------------------------------------------------------------
    wire        core_done;
    wire [15:0] core_rd_data;

    {core_module_name} #(
        .MAX_N     ({MAXn}),
        .ADDR_WIDTH({aw})
    ) core (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .done         (core_done),

        // Load: bit-reversed address, data packed into 24-bit unified format
        // We always load as FP8 (upper 16 bits of 24-bit word)
        .ext_wr_en    (load_en),
        .ext_wr_addr  (load_addr_rev),
        .ext_wr_data  ({{ load_data, 8'h00 }}),  // FP8 in [23:8], FP4=0 in [7:0]

        // Unload
        .ext_reading  (unload_en),
        .ext_rd_addr  ({unload_addr_expr}),
        .ext_rd_data  (core_rd_data),

        .ext_bank_sel (bank_sel)
    );

    assign unload_data = core_rd_data;

    // -------------------------------------------------------------------------
    // done register and bank_sel management
    //
    // bank_sel state machine:
    //   Loading phase : 1'b1  → memory writes go to bank0 (stage 0 reads bank0)
    //   FFT running   : managed by core internally
    //   FFT done      : ns%2  → holds result-bank pointer for unload
    //   Next load     : restored to 1'b1 when load_en fires (before start),
    //                   so new data always lands in bank0.
    //
    // -------------------------------------------------------------------------
    always @(posedge clk or negedge rst) begin
        if (!rst) begin
            done     <= 1'b0;
            bank_sel <= 1'b1;
        end else begin
            // Restore bank_sel for loading as soon as load_en fires.
            // This is before start, so it is safe to overwrite whatever
            // ns%2 value was left from the previous FFT.
            if (load_en && !start)
                bank_sel <= 1'b1;

            if (start) begin
                done     <= 1'b0;
                bank_sel <= 1'b1;
            end else if (core_done) begin
                done     <= 1'b1;
                // Point at the bank that holds the FFT result:
                //   odd  num_stages → bank 1  (bank_sel=1)
                //   even num_stages → bank 0  (bank_sel=0)
                bank_sel <= 1'b{ns % 2};
            end else if (!start && done) begin
                // Clear done flag. bank_sel intentionally left at ns%2
                // so the testbench can still read the result after done deasserts.
                // The load_en branch above will restore it before the next FFT.
                done <= 1'b0;
            end
        end
    end

endmodule
"""


# =============================================================================
# Quick smoke-test
# =============================================================================
if __name__ == "__main__":
    import os
    os.makedirs("./generated_designs", exist_ok=True)

    for fft_sz in [8, 16, 32]:
        gen = FFTTemplateGenerator(fft_size=fft_sz)
        ns  = gen.num_stages
        # Alternating FP4/FP8 per stage as a test chromosome
        chrom = []
        for s in range(ns):
            chrom += [s % 2, (s + 1) % 2]
        print(f"\n--- FFT-{fft_sz}  chromosome: {chrom} ---")
        core_f, top_f = gen.generate_verilog(
            chrom,
            f"./generated_designs/mixed_fft_{fft_sz}_test.v"
        )
        print(f"  Core : {core_f}")
        print(f"  Top  : {top_f}")
