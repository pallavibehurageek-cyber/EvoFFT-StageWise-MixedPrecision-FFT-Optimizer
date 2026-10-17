// Wrapper modules to choose the precision

module butterfly_wrapper #(
    parameter MULT_PRECISION = 0, // 0 for FP4, 1 for FP8
    parameter ADD_PRECISION  = 0  // 0 for FP4, 1 for FP8
)(
    input  [23:0] A, B,       // 24-bit unified format inputs
    input  [15:0] W,          // twiddle factor (16-bit FP8, [7:0] used for FP4)
    output [15:0] X, Y,       // 16-bit outputs (FP8 full / FP4 zero-padded to [7:0])
    output        output_is_fp8
);

    // Internal wires for each precision path
    wire [15:0] X_fp4_path, Y_fp4_path;   // FP4 result, zero-padded to 16 bits
    wire [15:0] X_fp8_path, Y_fp8_path;   // FP8 full 16-bit result
    wire [7:0]  X_fp4_raw,  Y_fp4_raw;    // Raw 8-bit FP4 complex output
    wire [15:0] X_mixed_84, Y_mixed_84;   // FP4 mul / FP8 add  (8add_4mul unit)
    wire [7:0]  X_mixed_48, Y_mixed_48;   // FP8 mul / FP4 add  (4add_8mul unit) raw

    generate
        if (MULT_PRECISION == 0 && ADD_PRECISION == 0) begin : USE_PURE_FP4
            fp4_butterfly_generation_unit fp4_butterfly_inst (
                .A(A[7:0]),
                .B(B[7:0]),
                .W(W),
                .X(X_fp4_raw),
                .Y(Y_fp4_raw)
            );
            assign X_fp4_path = {8'h00, X_fp4_raw};
            assign Y_fp4_path = {8'h00, Y_fp4_raw};
        end else begin
            assign X_fp4_path = 16'h0000;
            assign Y_fp4_path = 16'h0000;
            assign X_fp4_raw  = 8'h00;
            assign Y_fp4_raw  = 8'h00;
        end

        if (MULT_PRECISION == 1 && ADD_PRECISION == 1) begin : USE_PURE_FP8
            fp8_butterfly_generation_unit fp8_butterfly_inst (
                .A(A[23:8]),
                .B(B[23:8]),
                .W(W),
                .X(X_fp8_path),
                .Y(Y_fp8_path)
            );
        end else begin
            assign X_fp8_path = 16'h0000;
            assign Y_fp8_path = 16'h0000;
        end

        if (MULT_PRECISION == 0 && ADD_PRECISION == 1) begin : USE_FP8add_FP4mul
            // 4-bit multiplier, 8-bit adder: A is FP8 (16-bit), B is FP4 (8-bit)
            butterfly_generation_unit_8add_4mul fp4mul_fp8add_inst (
                .A(A[23:8]),
                .B(B[7:0]),
                .W(W),
                .X(X_mixed_84),
                .Y(Y_mixed_84)
            );
        end else begin
            assign X_mixed_84 = 16'h0000;
            assign Y_mixed_84 = 16'h0000;
        end

        if (MULT_PRECISION == 1 && ADD_PRECISION == 0) begin : USE_FP8mul_FP4add
            // 8-bit multiplier, 4-bit adder: A is FP4 (8-bit), B is FP8 (16-bit)
            butterfly_generation_unit_4add_8mul fp8mul_fp4add_inst (
                .A(A[7:0]),
                .B(B[23:8]),
                .W(W),
                .X(X_mixed_48),
                .Y(Y_mixed_48)
            );
        end else begin
            assign X_mixed_48 = 8'h00;
            assign Y_mixed_48 = 8'h00;
        end
    endgenerate

    // Select output based on precision parameters (elaboration-time constants)
    assign X = (MULT_PRECISION == 0 && ADD_PRECISION == 0) ? X_fp4_path  :
               (MULT_PRECISION == 1 && ADD_PRECISION == 1) ? X_fp8_path  :
               (MULT_PRECISION == 0 && ADD_PRECISION == 1) ? X_mixed_84  :
                                                             {8'h00, X_mixed_48};

    assign Y = (MULT_PRECISION == 0 && ADD_PRECISION == 0) ? Y_fp4_path  :
               (MULT_PRECISION == 1 && ADD_PRECISION == 1) ? Y_fp8_path  :
               (MULT_PRECISION == 0 && ADD_PRECISION == 1) ? Y_mixed_84  :
                                                             {8'h00, Y_mixed_48};

    // ADD_PRECISION determines whether the output is FP8 width
    assign output_is_fp8 = (ADD_PRECISION == 1) ? 1'b1 : 1'b0;

endmodule


module cmul_wrapper #(
    parameter PRECISION = 0  // 0 = FP4, 1 = FP8
)(
    input  [7:0] a,
    input  [7:0] b,
    input  [7:0] c,
    input  [7:0] d,
    output [7:0] out_real,
    output [7:0] out_imag
);
    generate
        if (PRECISION == 0) begin : USE_FP4
            wire [3:0] r4, i4;
            fp4_cmul inst_fp4 (
                .a(a[3:0]),
                .b(b[3:0]),
                .c(c[3:0]),
                .d(d[3:0]),
                .out_real(r4),
                .out_imag(i4)
            );
            assign out_real = {4'b0000, r4};
            assign out_imag = {4'b0000, i4};
        end else begin : USE_FP8
            fp8_cmul inst_fp8 (
                .a(a),
                .b(b),
                .c(c),
                .d(d),
                .out_real(out_real),
                .out_imag(out_imag)
            );
        end
    endgenerate
endmodule
