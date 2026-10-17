module fp4_add_sub(
    //module for addition and subtraction of two FP4 numbers
    //format: [sign][exp:2bits][mantissa:1bit]
    //normal numbers: value = (-1)^sign × 1.mantissa × 2^(exp-1)
    input [3:0] a,
    input [3:0] b,
    input sub, //sub=0 for addition, sub=1 for subtraction
    output [3:0] out
);

    //unpack the input numbers into sign, exponent, mantissa
    wire sign_a = a[3];
    wire [1:0] exp_a = a[2:1];
    wire mant_a = a[0];
    wire sign_b = b[3];
    wire [1:0] exp_b = b[2:1];
    wire mant_b = b[0];
    
    //if we're subtracting, flip the sign of b
    //this turns subtraction into addition with opposite sign
    wire sign_b_eff = sub ? ~sign_b : sign_b;
    
    //figure out which number has larger magnitude
    //compare exponents first, then mantissas if exponents are equal
    wire a_larger = (exp_a > exp_b) || 
                    ((exp_a == exp_b) && (mant_a >= mant_b));
    
    //assign larger and smaller numbers
    //we always align the smaller to the larger
    wire sign_l = a_larger ? sign_a : sign_b_eff;
    wire [1:0] exp_l = a_larger ? exp_a : exp_b;
    wire mant_l = a_larger ? mant_a : mant_b;
    wire sign_s = a_larger ? sign_b_eff : sign_a;
    wire [1:0] exp_s = a_larger ? exp_b : exp_a;
    wire mant_s = a_larger ? mant_b : mant_a;
    
    //calculate how much we need to shift the smaller number
    wire [1:0] exp_diff = exp_l - exp_s;
    
    wire hidden_l = (exp_l != 2'b00);  //hidden bit is 1 for normal, 0 for subnormal
    wire hidden_s = (exp_s != 2'b00);
    wire [2:0] sig_l = {1'b0, hidden_l, mant_l};
    wire [2:0] sig_s_unaligned = {1'b0, hidden_s, mant_s};
    
    wire [2:0] sig_s = (exp_diff == 2'd0) ? sig_s_unaligned :
                       (exp_diff == 2'd1) ? {1'b0, sig_s_unaligned[2:1]} :
                       (exp_diff == 2'd2) ? {2'b00, sig_s_unaligned[2]} :
                       3'b000;
    
    wire do_sub = (sign_l != sign_s);
    
    wire [3:0] sig_result_raw = do_sub ? 
                                (sig_l - sig_s) : 
                                (sig_l + sig_s);
    
    reg [1:0] exp_norm;
    reg [2:0] sig_norm;
    reg sign_out;
    
    always @(*) begin
        sign_out = sign_l;
        
        if (sig_result_raw[3]) begin
            sig_norm = {1'b0, sig_result_raw[3:2]};  //shift right by 2
            exp_norm = exp_l + 2'd2;                  //increment exponent by 2
        end
        else if (sig_result_raw[2]) begin
            sig_norm = {1'b0, sig_result_raw[2:1]};  //shift right by 1
            exp_norm = exp_l + 2'd1;                  //increment exponent by 1
        end
        else if (sig_result_raw[1]) begin
            sig_norm = {1'b0, sig_result_raw[1:0]};  //take as is
            if (exp_l == 2'b00) begin
                exp_norm = 2'b01;  //transition from subnormal to normal
            end else begin
                exp_norm = exp_l;  //exponent stays same
            end
        end
        else if (sig_result_raw[0]) begin
            if (exp_l <= 2'b01) begin
                // stays subnormal — place in correct subnormal slot
                sig_norm = {2'b00, sig_result_raw[0]};
                exp_norm = 2'b00;
            end else begin
                // normalize: 0001 → 1.0 × 2^(exp_l - 3)
                sig_norm = {1'b0, sig_result_raw[0], 1'b0};  // hidden bit position
                exp_norm = exp_l - 2'd3;
            end
        end
        else begin
            sig_norm = 3'b000;
            exp_norm = 2'b00;
            sign_out = 1'b0;  //zero is always positive
        end
    end
    
    wire mant_out = sig_norm[0];
    
    reg [3:0] result;
    always @(*) begin
        if (sig_norm == 3'b000) begin
            result = 4'b0000;
        end
        else if (exp_norm[1] && exp_norm[0] && mant_out) begin
            result = {sign_out, 2'b11, 1'b1};
        end
        else if (exp_norm == 2'b00) begin
            result = 4'b0000;
        end
        else begin
            result = {sign_out, exp_norm, mant_out};
        end
    end
    
    assign out = result;

endmodule

module fp4_complex_add_sub(
    input [7:0] a,
    input [7:0] b,
    input sub, //sub=0 for addition, sub=1 for subtraction
    output [7:0] out
);
    wire [3:0] a_real = a[7:4];
    wire [3:0] a_imag = a[3:0];
    wire [3:0] b_real = b[7:4];
    wire [3:0] b_imag = b[3:0];
    
    wire [3:0] out_real;
    wire [3:0] out_imag;
    
    fp4_add_sub adder_real (
        .a(a_real), .b(b_real), .sub(sub), .out(out_real)
    );
    
    fp4_add_sub adder_imag (
        .a(a_imag), .b(b_imag), .sub(sub), .out(out_imag)
    );
    
    assign out = {out_real, out_imag};

endmodule

module fp8_add_sub(
    // format: [sign][exp:4bits][mantissa:3bits]
    // bias = 7
    input [7:0] a,
    input [7:0] b,
    input sub, //sub=0 for addition, sub=1 for subtraction
    output [7:0] out
);
    
    wire sign_a = a[7];
    wire [3:0] exp_a = a[6:3];
    wire [2:0] mant_a = a[2:0];
    
    wire sign_b = b[7];
    wire [3:0] exp_b = b[6:3];
    wire [2:0] mant_b = b[2:0];

    wire sign_b_eff = sub ? ~sign_b : sign_b;

    wire a_larger = (exp_a > exp_b) || 
                    ((exp_a == exp_b) && (mant_a >= mant_b));

    wire sign_l = a_larger ? sign_a : sign_b_eff;
    wire [3:0] exp_l = a_larger ? exp_a : exp_b;
    wire [2:0] mant_l = a_larger ? mant_a : mant_b;
    wire sign_s = a_larger ? sign_b_eff : sign_a;
    wire [3:0] exp_s = a_larger ? exp_b : exp_a;
    wire [2:0] mant_s = a_larger ? mant_b : mant_a;
    
    wire [3:0] exp_diff = exp_l - exp_s;

    wire hidden_l = (exp_l != 4'b0000); 
    wire hidden_s = (exp_s != 4'b0000);
    
    // We create a 7-bit value: [overflow_bits:2][hidden_bit:1][mantissa_bits:3][guard:1]
    wire [6:0] sig_l = {2'b00, hidden_l, mant_l, 1'b0};
    wire [6:0] sig_s_unaligned = {2'b00, hidden_s, mant_s, 1'b0};
    
    wire [6:0] sig_s = (exp_diff == 4'd0) ? sig_s_unaligned :
                       (exp_diff == 4'd1) ? {1'b0, sig_s_unaligned[6:1]} :
                       (exp_diff == 4'd2) ? {2'b00, sig_s_unaligned[6:2]} :
                       (exp_diff == 4'd3) ? {3'b000, sig_s_unaligned[6:3]} :
                       (exp_diff == 4'd4) ? {4'b0000, sig_s_unaligned[6:4]} :
                       (exp_diff == 4'd5) ? {5'b00000, sig_s_unaligned[6:5]} :
                       (exp_diff == 4'd6) ? {6'b000000, sig_s_unaligned[6]} :
                       7'b0000000;
    
    wire do_sub = (sign_l != sign_s);
    
    wire [7:0] sig_result_raw = do_sub ? 
                                ({1'b0, sig_l} - {1'b0, sig_s}) : 
                                ({1'b0, sig_l} + {1'b0, sig_s});
    
    reg [3:0] exp_norm;
    reg [6:0] sig_norm;
    reg sign_out;
    
    always @(*) begin
        sign_out = sign_l;
        
        if (sig_result_raw[6]) begin 
            sig_norm = {2'b00, sig_result_raw[6:2]};
            exp_norm = exp_l + 4'd2;
        end
        else if (sig_result_raw[5]) begin 
            sig_norm = {2'b00, sig_result_raw[5:1]};
            exp_norm = exp_l + 4'd1;
        end
        else if (sig_result_raw[4]) begin 
            sig_norm = {2'b00, sig_result_raw[4:0]};
            if (exp_l == 4'b0000) exp_norm = 4'b0001; 
            else exp_norm = exp_l;
        end
        else if (sig_result_raw[3]) begin 
            if (exp_l == 4'b0001 || exp_l == 4'b0000) begin
                sig_norm = {2'b00, sig_result_raw[4:0]}; 
                exp_norm = 4'b0000;
            end else begin
                sig_norm = {2'b00, sig_result_raw[3:0], 1'b0};
                exp_norm = exp_l - 4'd1;
            end
        end
        else if (sig_result_raw[2]) begin 
            if (exp_l <= 4'd2) begin
                sig_norm = {2'b00, sig_result_raw[4:0]};
                exp_norm = 4'b0000;
            end else begin
                sig_norm = {2'b00, sig_result_raw[2:0], 2'b00};
                exp_norm = exp_l - 4'd2;
            end
        end
        else if (sig_result_raw[1]) begin 
            if (exp_l <= 4'd3) begin
                sig_norm = {2'b00, sig_result_raw[4:0]};
                exp_norm = 4'b0000;
            end else begin
                sig_norm = {2'b00, sig_result_raw[1:0], 3'b000};
                exp_norm = exp_l - 4'd3;
            end
        end
        else if (sig_result_raw[0]) begin 
            if (exp_l <= 4'd4) begin
                sig_norm = {2'b00, sig_result_raw[4:0]};
                exp_norm = 4'b0000;
            end else begin
                sig_norm = {2'b00, sig_result_raw[0], 4'b0000};
                exp_norm = exp_l - 4'd4;
            end
        end
        else begin
            sig_norm = 7'd0;
            exp_norm = 4'd0;
            sign_out = 1'b0;
        end
    end
    
    // sig_norm map: [ov:2][hidden:1][mant:3][guard:1]
    wire guard      = sig_norm[0];
    wire round_up   = guard & sig_norm[1];  // round-to-nearest-even
    wire [2:0] mant_rounded = sig_norm[3:1] + round_up;
    
    // Handle overflow: if mantissa wraps, increment exponent
    wire mant_ovf = &sig_norm[3:1] & round_up;  // all 1s + 1 = overflow
    wire [3:0] exp_final  = mant_ovf ? exp_norm + 1 : exp_norm;
    wire [2:0] mant_out = mant_rounded;  // wraps naturally to 000 on overflow
    
    reg [7:0] result;
    always @(*) begin
        if (sig_norm == 7'd0) begin
            result = 8'b00000000;
        end
        else if (exp_final >= 4'd15) begin
            result = {sign_out, 4'd15, 3'd6};  // E4M3 saturation (0_1111_110)
        end
        else begin
            result = {sign_out, exp_final, mant_out};
        end
    end
    
    assign out = result;

endmodule

module fp8_complex_add_sub(
    input [15:0] a,
    input [15:0] b,
    input sub, //sub=0 for addition, sub=1 for subtraction
    output [15:0] out
);
    wire [7:0] a_real = a[15:8];
    wire [7:0] a_imag = a[7:0];
    wire [7:0] b_real = b[15:8];
    wire [7:0] b_imag = b[7:0];
    
    wire [7:0] out_real;
    wire [7:0] out_imag;
    
    fp8_add_sub adder_real (
        .a(a_real), .b(b_real), .sub(sub), .out(out_real)
    );
    
    fp8_add_sub adder_imag (
        .a(a_imag), .b(b_imag), .sub(sub), .out(out_imag)
    );
    
    assign out = {out_real, out_imag};
endmodule
