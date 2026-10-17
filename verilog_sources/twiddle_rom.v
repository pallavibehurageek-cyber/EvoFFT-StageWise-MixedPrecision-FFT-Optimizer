// Unified Twiddle ROM with 24-bit format
// Format: [23:16] FP8 Real, [15:8] FP8 Imag, [7:4] FP4 Real, [3:0] FP4 Imag
module twiddle_factor_unified#(
    parameter MAX_N = 1024,
    parameter ADDR_WIDTH = $clog2(MAX_N) + 1
)(
    input [ADDR_WIDTH-1:0] k,   // Index k
    input [ADDR_WIDTH-1:0] n,     // Current FFT size N
    input PRECISION,
    output reg [15:0] twiddle_out
);

    // --------------------------------------------------------
    // 1. ROM Declaration (512 entries, 24-bit each)
    // --------------------------------------------------------
    // Format: [23:16] FP8 Real, [15:8] FP8 Imag, [7:4] FP4 Real, [3:0] FP4 Imag
    reg [23:0] rom [0:511];

    initial begin
        // Read the binary file containing unified 24-bit format
        $readmemb("twiddles_1024.txt", rom);
    end

    // --------------------------------------------------------
    // 2. Dynamic Scaling Logic
    // --------------------------------------------------------
    reg [ADDR_WIDTH-1:0] scaled_k;

    always @(*) begin
        case (n)
            1024: scaled_k = k;
            512:  scaled_k = {k, 1'b0};      // k * 2
            256:  scaled_k = {k, 2'b00};     // k * 4
            128:  scaled_k = {k, 3'b000};    // k * 8
            64:   scaled_k = {k, 4'b0000};   // k * 16
            32:   scaled_k = {k, 5'b00000};  // k * 32
            16:   scaled_k = {k, 6'b000000}; // k * 64
            8:    scaled_k = {k, 7'b0000000};// k * 128
            4:    scaled_k = {k, 8'b00000000};// k * 256
            2:    scaled_k = {k, 9'b000000000};// k * 512
            default: scaled_k = 11'd0;
        endcase
    end

    // --------------------------------------------------------
    // 3. Symmetry Logic & Fetch
    // --------------------------------------------------------
    reg use_conjugate;
    reg [9:0] rom_addr; // Address within the 0-511 block
    reg is_midpoint;

    always @(*) begin
        is_midpoint = 1'b0;

        if (scaled_k == 512) begin
            // 180 degrees (Index 512) is a boundary case
            is_midpoint = 1'b1;
            rom_addr = 0; 
            use_conjugate = 1'b0;
        end 
        else if (scaled_k > 511) begin
            // Second half (180 < angle < 360) -> Symmetry
            rom_addr = 1024 - scaled_k;
            use_conjugate = 1'b1;
        end 
        else begin
            // First half (0 <= angle < 180)
            rom_addr = scaled_k;
            use_conjugate = 1'b0;
        end
    end

    // --------------------------------------------------------
    // 4. Output Generation with Precision Selection
    // --------------------------------------------------------
    reg [23:0] raw_data;
    reg [15:0] selected_data;

    always @(*) begin
        if (is_midpoint) begin
            // Hardcoded -1.0 value
            if (PRECISION == 1) 
                selected_data = 16'hB800; // FP8 (-1.0, real=-1, imag=0)
            else 
                selected_data = 16'h00A0; // FP4 (-1.0, real=-1, imag=0)
        end else begin
            // Read from ROM
            raw_data = rom[rom_addr];
            
            // Select FP8 or FP4 based on PRECISION parameter
            if (PRECISION == 1) begin
                // FP8: Extract bits [23:8]
                selected_data = raw_data[23:8];
            end else begin
                // FP4: Extract bits [7:0] and extend to 16 bits
                selected_data = {8'h00, raw_data[7:0]};
            end
        end

        twiddle_out = selected_data;

        // Apply Conjugate (Flip sign of imaginary part)
        if (use_conjugate) begin
            if (PRECISION == 1) begin
                // FP8: [7:0] is Imaginary
                if (twiddle_out[7:0] != 8'h00) 
                    twiddle_out[7:0] = {~twiddle_out[7], twiddle_out[6:0]};
            end else begin
                // FP4: [3:0] is Imaginary
                if (twiddle_out[3:0] != 4'h0) 
                    twiddle_out[3:0] = {~twiddle_out[3], twiddle_out[2:0]};
            end
        end
    end

endmodule
