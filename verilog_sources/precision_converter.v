// Mixed-Precision Butterfly Module with Independent Multiplier and Adder Precision
// Supports all 4 combinations:
// - FP4 mult + FP4 add
// - FP4 mult + FP8 add
// - FP8 mult + FP4 add  
// - FP8 mult + FP8 add

// Helper Module: FP4 to FP8 Converter
// FP4 E2M1: [sign:1][exp:2][mant:1]  bias = 1
// FP8 E4M3: [sign:1][exp:4][mant:3]  bias = 7
// Conversion: true_exp = exp_fp4 - 1  →  exp_fp8 = true_exp + 7 = exp_fp4 + 6
module fp4_to_fp8_converter(
    input  [3:0] fp4_in,
    output [7:0] fp8_out
);
    wire        sign     = fp4_in[3];
    wire [1:0]  exp_fp4  = fp4_in[2:1];
    wire        mant_fp4 = fp4_in[0];

    reg [3:0] exp_fp8;
    reg [2:0] mant_fp8;

    always @(*) begin
        if (exp_fp4 == 2'b00) begin
            if (mant_fp4 == 1'b0) begin
                // +/- Zero
                exp_fp8  = 4'b0000;
                mant_fp8 = 3'b000;
            end else begin
                // FP4 subnormal: value = 0.1 × 2^(1-1) = 0.5 = 2^(-1)
                // Represent in FP8 as normal: 1.000 × 2^(-1)
                // exp_fp8 = -1 + 7 = 6
                exp_fp8  = 4'd6;
                mant_fp8 = 3'b000;
            end
        end else begin
            // Normal FP4: value = 1.mant × 2^(exp_fp4 - 1)
            // exp_fp8 = (exp_fp4 - 1) + 7 = exp_fp4 + 6
            exp_fp8  = {2'b00, exp_fp4} + 4'd6;
            // Extend mantissa: FP4 has 1 bit, FP8 has 3 bits
            // Zero-pad the lower 2 bits — this is exact (no information lost)
            mant_fp8 = {mant_fp4, 2'b00};
        end
    end

    assign fp8_out = {sign, exp_fp8, mant_fp8};
endmodule


// Helper Module: FP8 to FP4 Converter (with correct rounding)
// FP8 E4M3: [sign:1][exp:4][mant:3]  bias = 7
// FP4 E2M1: [sign:1][exp:2][mant:1]  bias = 1
// Conversion: true_exp = exp_fp8 - 7  →  exp_fp4 = true_exp + 1 = exp_fp8 - 6
// FP4 representable range: exp_fp8 in [6, 9]  (true_exp in [-1, 2])
module fp8_to_fp4_converter(
    input  [7:0] fp8_in,
    output [3:0] fp4_out
);
    wire        sign     = fp8_in[7];
    wire [3:0]  exp_fp8  = fp8_in[6:3];
    wire [2:0]  mant_fp8 = fp8_in[2:0];

    // --- Exponent conversion (module-level wire, full 4-bit subtraction) ---
    // Only valid when exp_fp8 in [6,9]; boundary cases handled in always block
    wire [3:0] exp_fp4_full = exp_fp8 - 4'd6;

    // --- Rounding: dropping 2 mantissa bits (3 → 1) ---
    // kept bit  : mant_fp8[2]
    // round bit : mant_fp8[1]  (first dropped bit)
    // sticky bit: mant_fp8[0]  (OR of remaining dropped bits)
    // Round-to-nearest-even: round up when round=1 AND (sticky=1 OR kept=1)
    wire round_bit       = mant_fp8[1];
    wire sticky_bit      = mant_fp8[0];
    wire round_up        = round_bit & (sticky_bit | mant_fp8[2]);
    wire mant_rounded    = mant_fp8[2] + round_up;   // 1-bit result; wraps 1→0 on overflow
    wire mant_ovf        = mant_fp8[2] & round_up;   // 1 only when 1+1 overflows

    reg [1:0] exp_fp4;
    reg       mant_fp4;

    always @(*) begin
        if (exp_fp8 == 4'b0000) begin
            // Zero or FP8 subnormal — too small for FP4, flush to zero
            exp_fp4  = 2'b00;
            mant_fp4 = 1'b0;
        end else if (exp_fp8 < 4'd6) begin
            // Underflow: magnitude too small for FP4 range → zero
            exp_fp4  = 2'b00;
            mant_fp4 = 1'b0;
        end else if (exp_fp8 > 4'd9) begin
            // Overflow: magnitude too large for FP4 range → saturate to max FP4
            exp_fp4  = 2'b11;
            mant_fp4 = 1'b1;
        end else begin
            // Normal range: exp_fp8 in [6, 9]
            // exp_fp4 = exp_fp8 - 6  (full 4-bit subtract, take lower 2 bits)
            exp_fp4  = exp_fp4_full[1:0];
            mant_fp4 = mant_rounded;

            // Mantissa overflow: rounding caused 1 → 0 wrap, increment exponent
            if (mant_ovf) begin
                exp_fp4 = exp_fp4_full[1:0] + 2'd1;
                // If this pushes exp_fp4 above 2'b11, saturate
                if (exp_fp4_full[1:0] == 2'b11) begin
                    exp_fp4  = 2'b11;
                    mant_fp4 = 1'b1;
                end
            end
        end
    end

    assign fp4_out = {sign, exp_fp4, mant_fp4};
endmodule


// Helper Module: Complex FP4 to FP8 Converter
module complex_fp4_to_fp8(
    input  [7:0]  complex_fp4,   // {real[3:0], imag[3:0]}
    output [15:0] complex_fp8    // {real[7:0], imag[7:0]}
);
    fp4_to_fp8_converter conv_real(
        .fp4_in  (complex_fp4[7:4]),
        .fp8_out (complex_fp8[15:8])
    );
    fp4_to_fp8_converter conv_imag(
        .fp4_in  (complex_fp4[3:0]),
        .fp8_out (complex_fp8[7:0])
    );
endmodule


// Helper Module: Complex FP8 to FP4 Converter
module complex_fp8_to_fp4(
    input  [15:0] complex_fp8,   // {real[7:0], imag[7:0]}
    output [7:0]  complex_fp4    // {real[3:0], imag[3:0]}
);
    fp8_to_fp4_converter conv_real(
        .fp8_in  (complex_fp8[15:8]),
        .fp4_out (complex_fp4[7:4])
    );
    fp8_to_fp4_converter conv_imag(
        .fp8_in  (complex_fp8[7:0]),
        .fp4_out (complex_fp4[3:0])
    );
endmodule
