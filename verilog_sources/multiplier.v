// Module for complex multiplication
// Main module name is fp4_cmul. Uses 2 helper modules : fp4_mul(multiplying 2 FP4 numbers) and fp4_add_sub(adding or subtracting 2 FP4 numbers)
//E2M1 format is used: 1 sign bit(MSB bit = 3), 2 exponent bits(bits 2 and 1), 1 mantissa bit(LSB bit = 1)

module fp4_mul(
    input [3:0] a,
    input [3:0] b,
    output [3:0] out
);
    //Unpacking the FP4 numbers to extract the sign bit, exponent bits, mantissa bit
    wire sign_a = a[3], sign_b = b[3];
    wire [1:0] exp_a = a[2:1], exp_b = b[2:1];
    wire mant_a = a[0], mant_b = b[0];

    //Sign bit of the output
    wire sign_out = sign_a ^ sign_b;

    //Detecting whether the inputs are 0
    wire a_zero = (exp_a == 2'b00 && mant_a == 1'b0);
    wire b_zero = (exp_b == 2'b00 && mant_b == 1'b0);

    //Temporary exponent bits
    wire [2:0] exp_sum = {1'b0, exp_a} + {1'b0, exp_b};
    wire signed [3:0] exp_temp = $signed({1'b0, exp_sum}) - 4'sd1;

    //Normalization and Rounding off
    wire hidden_a = (exp_a != 2'b00);
    wire hidden_b = (exp_b != 2'b00);
    wire [2:0] sig_a = {1'b0, hidden_a, mant_a};
    wire [2:0] sig_b = {1'b0, hidden_b, mant_b};
    wire [4:0] prod = sig_a * sig_b;
    wire need_norm = (prod >= 5'd8);
    wire [4:0] prod_norm = (need_norm) ? (prod >> 1) : prod;
    wire signed [3:0] exp_norm = (need_norm) ? (exp_temp + 1) : exp_temp;
    wire mant_out = prod_norm[1];  // correct round-to-nearest for all cases
    // prod_norm[1] is the MSB of the result mantissa after normalization
    // prod_norm[0] would be the round bit — add if you want R-T-N-E:
    // wire round_up = prod_norm[0] & (prod_norm[1]);  // tie-break to even
    // wire mant_out = prod_norm[1] + round_up;        // with overflow check

    //Preparing the output
    reg [3:0] output_reg;
    always @(*) begin
        if(a_zero || b_zero || exp_norm<0) begin
            output_reg = 4'b0000;
        end
        else if(exp_norm > 2'b11 || exp_norm == 2'b11 && mant_out) begin
            output_reg = {sign_out, 2'b11, 1'b1};
        end
        else begin
            output_reg = {sign_out, exp_norm[1:0], mant_out};
        end
    end
    assign out = output_reg;

endmodule

// Let's consider 2 complex numbers z1 = a + jb and z2 = c + jd
// Inputs have been named following the above nomenclature
// z1*z2 = (ac - bd) + j(ad + bc) => Re(z1*z2) = ac-bd and Im(z1*z2) = ad + bc
// Clearly we require 4 multipliers to compute each of the products above and 2 adder_subtractor modules
// The below fp4_cmul module handles complex multiplication
module fp4_cmul (
    input [3:0] a,
    input [3:0] b,
    input [3:0] c,
    input [3:0] d,
    output [3:0] out_real,
    output [3:0] out_imag
);
    wire [3:0] ac, bd, ad, bc;
    fp4_mul m1(.a(a), .b(c), .out(ac));
    fp4_mul m2(.a(b), .b(d), .out(bd));
    fp4_mul m3(.a(a), .b(d), .out(ad));
    fp4_mul m4(.a(b), .b(c), .out(bc));

    wire [3:0] res_real;
    wire [3:0] res_imag;
    fp4_add_sub s1(.a(ac), .b(bd), .sub(1'b1), .out(res_real));
    fp4_add_sub a1(.a(ad), .b(bc), .sub(1'b0), .out(res_imag));
    assign out_real = res_real;
    assign out_imag = res_imag;
endmodule


// Module for complex multiplication

// Main module name is fp8_cmul. Uses 2 helper modules : fp8_mul(multiplying 2 FP8 numbers) and fp8_add_sub(adding or subtracting 2 FP8 numbers)

//E4M3 format is used: 1 Sign Bit (MSB Bit 7), 4 Exponent Bit(Bits 6 to 3), 3 mantissa bits(Bits 2 to 0)

module fp8_mul(
    input [7:0] a,
    input [7:0] b,
    output [7:0] out
);
    //Unpacking the FP8 numbers to extract the sign bit, exponent bits, mantissa bit
    wire sign_a = a[7], sign_b = b[7];
    wire [3:0] exp_a = a[6:3], exp_b = b[6:3];
    wire [2:0] mant_a = a[2:0], mant_b = b[2:0];

    //Sign bit of the output
    wire sign_out = sign_a ^ sign_b;

    //Detecting whether the inputs are 0
    wire a_zero = (exp_a == 4'b0000 && mant_a == 3'b000);
    wire b_zero = (exp_b == 4'b0000 && mant_b == 3'b000);

    //Temporary exponent bits
    wire [4:0] exp_sum = {1'b0, exp_a} + {1'b0, exp_b};
    wire signed [5:0] exp_temp = $signed({1'b0, exp_sum}) - 4'sd7;

    //Normalization and Rounding off
    wire hidden_a = (exp_a != 4'b0000);
    wire hidden_b = (exp_b != 4'b0000);
    wire [4:0] sig_a = {1'b0, hidden_a, mant_a};
    wire [4:0] sig_b = {1'b0, hidden_b, mant_b};
    wire [10:0] prod = sig_a * sig_b;
    wire need_norm = (prod >= 8'd128);
    wire [10:0] prod_norm = (need_norm) ? (prod >> 1) : prod;
    wire signed [5:0] exp_norm = (need_norm) ? (exp_temp + 1) : exp_temp;
    wire [2:0] mant_out = prod_norm[5:3];

    //Preparing the output
    reg [7:0] output_reg;
    always @(*) begin
        if(a_zero || b_zero || exp_norm<0) begin
            output_reg = 8'h00;
        end
        else if(exp_norm > 4'b1111 || exp_norm == 4'b1111 && mant_out) begin
            output_reg = {sign_out, 4'b1111, 3'b111};
        end
        else begin
            output_reg = {sign_out, exp_norm[3:0], mant_out};
        end
    end
    assign out = output_reg;

endmodule

// Let's consider 2 complex numbers z1 = a + jb and z2 = c + jd
// Inputs have been named following the above nomenclature
// z1*z2 = (ac - bd) + j(ad + bc) => Re(z1*z2) = ac-bd and Im(z1*z2) = ad + bc
// Clearly we require 4 multipliers to compute each of the products above and 2 adder_subtractor modules
// The below fp8_cmul module handles complex multiplication
module fp8_cmul (
    input [7:0] a,
    input [7:0] b,
    input [7:0] c,
    input [7:0] d,
    output [7:0] out_real,
    output [7:0] out_imag
);
    wire [7:0] ac, bd, ad, bc;
    fp8_mul m1(.a(a), .b(c), .out(ac));
    fp8_mul m2(.a(b), .b(d), .out(bd));
    fp8_mul m3(.a(a), .b(d), .out(ad));
    fp8_mul m4(.a(b), .b(c), .out(bc));

    wire [7:0] res_real, res_imag;
    fp8_add_sub s1(.a(ac), .b(bd), .sub(1'b1), .out(res_real));
    fp8_add_sub a1(.a(ad), .b(bc), .sub(1'b0), .out(res_imag));
    assign out_real = res_real;
    assign out_imag = res_imag;
endmodule
