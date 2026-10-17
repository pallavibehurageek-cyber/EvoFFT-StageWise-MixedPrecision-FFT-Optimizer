// =============================================================================
// Mixed-Precision FFT Memory — BRAM-inferred version
//
// BRAM inference rules (Xilinx UG901):
//   - Array must be written on a clock edge
//   - Array must be read on a clock edge (synchronous read)
//   - No asynchronous / combinational reset on the array itself
//   - Width × Depth must fit a RAMB36 or RAMB18 primitive
//
// For n=1024, 24-bit word: 24 576 bits per bank → fits in one RAMB36E1 (36 Kb)
// Two banks (ping-pong) → 2× RAMB36, 0 LUTs for storage.
// =============================================================================

// -----------------------------------------------------------------------------
// Unified Mixed-Precision Memory (ping-pong, 24-bit word)
// Format: [23:16] FP8 Real | [15:8] FP8 Imag | [7:4] FP4 Real | [3:0] FP4 Imag
// -----------------------------------------------------------------------------
module mixed_memory_unified #(
    parameter n          = 1024,
    parameter ADDR_WIDTH = $clog2(n) + 1
)(
    input  wire                  clk,
    input  wire                  rst,        // active-low; used only for output FF

    // bank_sel: 0 → read bank0 / write bank1 ; 1 → read bank1 / write bank0
    input  wire                  bank_sel,

    // Port 0 — read
    input  wire [ADDR_WIDTH-1:0] rd_addr_0,
    input  wire                  rd_precision_0,   // 0=FP4, 1=FP8
    output wire [15:0]           rd_data_0,

    // Port 1 — write
    input  wire                  wr_en_1,
    input  wire [ADDR_WIDTH-1:0] wr_addr_1,
    input  wire [23:0]           wr_data_1         // full 24-bit write
);

    // -------------------------------------------------------------------------
    // Memory arrays — NO reset loop so Vivado can infer BRAM
    // -------------------------------------------------------------------------
    (* ram_style = "block" *) reg [23:0] bank0_mem [0:n-1];
    (* ram_style = "block" *) reg [23:0] bank1_mem [0:n-1];

    // -------------------------------------------------------------------------
    // Write port (synchronous, no reset on array)
    // bank_sel=0 → reading bank0, so write to bank1, and vice-versa
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (wr_en_1) begin
            if (bank_sel == 1'b0)
                bank1_mem[wr_addr_1] <= wr_data_1;
            else
                bank0_mem[wr_addr_1] <= wr_data_1;
        end
    end

    // -------------------------------------------------------------------------
    // Read port — synchronous (1-cycle latency), no reset on array
    // Precision mux is pipelined one extra cycle to ease timing
    // -------------------------------------------------------------------------
    reg [23:0] rd_data_full;
    reg        rd_prec_d;
    reg [15:0] rd_data_reg;

    always @(posedge clk) begin
        // Stage 1: BRAM read (synchronous)
        if (bank_sel == 1'b0)
            rd_data_full <= bank0_mem[rd_addr_0];
        else
            rd_data_full <= bank1_mem[rd_addr_0];

        rd_prec_d <= rd_precision_0;   // delay precision select to match data

        // Stage 2: precision mux + output register (reset allowed on plain FF)
        if (!rst) begin
            rd_data_reg <= 16'b0;
        end else begin
            if (rd_prec_d)
                rd_data_reg <= rd_data_full[23:8];   // FP8: upper 16 bits
            else
                rd_data_reg <= {8'h00, rd_data_full[7:0]};  // FP4: lower 8 bits, zero-extended
        end
    end

    assign rd_data_0 = rd_data_reg;

endmodule


// -----------------------------------------------------------------------------
// FP4-only memory (8-bit word, ping-pong)
// Backward-compatible with fp4_fft_memory_reg interface
// -----------------------------------------------------------------------------
module fp4_fft_memory_reg #(
    parameter n          = 1024,
    parameter ADDR_WIDTH = $clog2(n)
)(
    input  wire                  clk,
    input  wire                  rst,

    input  wire                  bank_sel,

    input  wire [ADDR_WIDTH-1:0] rd_addr_0,
    output wire [7:0]            rd_data_0,

    input  wire                  wr_en_1,
    input  wire [ADDR_WIDTH-1:0] wr_addr_1,
    input  wire [7:0]            wr_data_1
);

    (* ram_style = "block" *) reg [7:0] bank0_mem [0:n-1];
    (* ram_style = "block" *) reg [7:0] bank1_mem [0:n-1];

    // Write
    always @(posedge clk) begin
        if (wr_en_1) begin
            if (bank_sel == 1'b0)
                bank1_mem[wr_addr_1] <= wr_data_1;
            else
                bank0_mem[wr_addr_1] <= wr_data_1;
        end
    end

    // Read
    reg [7:0] rd_data_reg;

    always @(posedge clk) begin
        if (!rst)
            rd_data_reg <= 8'b0;
        else begin
            if (bank_sel == 1'b0)
                rd_data_reg <= bank0_mem[rd_addr_0];
            else
                rd_data_reg <= bank1_mem[rd_addr_0];
        end
    end

    assign rd_data_0 = rd_data_reg;

endmodule


// -----------------------------------------------------------------------------
// FP8-only memory (16-bit word, ping-pong)
// Backward-compatible with fp8_fft_memory_reg interface
// -----------------------------------------------------------------------------
module fp8_fft_memory_reg #(
    parameter n          = 1024,
    parameter ADDR_WIDTH = $clog2(n)
)(
    input  wire                  clk,
    input  wire                  rst,

    input  wire                  bank_sel,

    input  wire [ADDR_WIDTH-1:0] rd_addr_0,
    output wire [15:0]           rd_data_0,

    input  wire                  wr_en_1,
    input  wire [ADDR_WIDTH-1:0] wr_addr_1,
    input  wire [15:0]           wr_data_1
);

    (* ram_style = "block" *) reg [15:0] bank0_mem [0:n-1];
    (* ram_style = "block" *) reg [15:0] bank1_mem [0:n-1];

    // Write
    always @(posedge clk) begin
        if (wr_en_1) begin
            if (bank_sel == 1'b0)
                bank1_mem[wr_addr_1] <= wr_data_1;
            else
                bank0_mem[wr_addr_1] <= wr_data_1;
        end
    end

    // Read
    reg [15:0] rd_data_reg;

    always @(posedge clk) begin
        if (!rst)
            rd_data_reg <= 16'b0;
        else begin
            if (bank_sel == 1'b0)
                rd_data_reg <= bank0_mem[rd_addr_0];
            else
                rd_data_reg <= bank1_mem[rd_addr_0];
        end
    end

    assign rd_data_0 = rd_data_reg;

endmodule
