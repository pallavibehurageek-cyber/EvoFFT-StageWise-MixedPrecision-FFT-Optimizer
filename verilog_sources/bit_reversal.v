module bit_reverse #(
    parameter MAX_N = 1024,
    parameter WIDTH = $clog2(MAX_N) + 1
)(
    input  [WIDTH-1:0] in,
    input  [WIDTH-1:0] N,  // runtime N value needed to shift bits
    output reg [WIDTH-1:0] out
);
    reg [WIDTH-1:0] log2_N;
    reg [WIDTH-1:0] temp_out;
    integer i;
    
    // Explicit 10-bit matching to prevent simulator case/shift evaluation bugs
    always @(*) begin
        if      (N == 11'd1024 || N == 11'd0) log2_N = 11'd10;
        else if (N == 11'd512) log2_N = 11'd9;
        else if (N == 11'd256) log2_N = 11'd8;
        else if (N == 11'd128) log2_N = 11'd7;
        else if (N == 11'd64)  log2_N = 11'd6;
        else if (N == 11'd32)  log2_N = 11'd5;
        else if (N == 11'd16)  log2_N = 11'd4;
        else if (N == 11'd8)   log2_N = 11'd3;
        else if (N == 11'd4)   log2_N = 11'd2;
        else if (N == 11'd2)   log2_N = 11'd1;
        else                   log2_N = 11'd3; // Failsafe
        
        // Full width bit reversal using a temporary variable
        temp_out = 0;
        for (i = 0; i < WIDTH; i = i + 1) begin
            temp_out[i] = in[WIDTH-1-i];
        end
        
        // Shift down to align the active bits to LSB
        out = temp_out >> (WIDTH - log2_N);
    end
endmodule
