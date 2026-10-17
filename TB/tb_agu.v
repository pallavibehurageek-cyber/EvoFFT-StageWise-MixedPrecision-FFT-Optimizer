
`timescale 1ns/1ps

module tb_agu;

    parameter MAX_N = 16;
    parameter ADDR_WIDTH = $clog2(MAX_N) + 1;

    reg clk;
    reg reset;
    reg start;
    reg next_step;
    reg [ADDR_WIDTH-1:0] N;

    wire [ADDR_WIDTH-1:0] idx_a;
    wire [ADDR_WIDTH-1:0] idx_b;
    wire [ADDR_WIDTH-1:0] k;
    wire done_stage;
    wire done_fft;
    wire [ADDR_WIDTH-1:0] curr_stage;
    wire [15:0] twiddle_output;

    integer errors;

    // DUT: Address Generation Unit
    dit_fft_agu_variable #(
        .MAX_N(MAX_N),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .N(N),
        .next_step(next_step),
        .idx_a(idx_a),
        .idx_b(idx_b),
        .k(k),
        .done_stage(done_stage),
        .done_fft(done_fft),
        .curr_stage(curr_stage),
        .twiddle_output(twiddle_output)
    );

    // 10 ns clock period
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // Advance the AGU by one butterfly
    task tick;
        begin
            @(negedge clk);
            next_step = 1'b1;

            @(posedge clk);
            #1;
            next_step = 1'b0;
            #1;
        end
    endtask

    // Test one FFT size
    task run_agu_case;
        input integer n_value;
        input integer stage_count;

        integer s;
        integer b;
        integer g;
        integer stride;
        integer groups;
        integer exp_a;
        integer exp_b;
        integer exp_k;
        reg last_in_stage;

        begin
            $display("");
            $display("Testing AGU with N = %0d", n_value);

            // Start a fresh FFT run
            @(negedge clk);
            N = n_value;
            next_step = 1'b0;
            start = 1'b1;

            @(posedge clk);
            #1;
            start = 1'b0;
            #1;

            // Visit every stage
            for (s = 0; s < stage_count; s = s + 1) begin

                stride = 1 << s;
                groups = n_value / (2 * stride);

                $display("  Stage %0d: stride=%0d, groups=%0d",
                         s, stride, groups);

                // The DUT iterates butterfly index first,
                // then group index.
                for (b = 0; b < stride; b = b + 1) begin
                    for (g = 0; g < groups; g = g + 1) begin

                        exp_a = g * 2 * stride + b;
                        exp_b = exp_a + stride;
                        exp_k = b * (n_value / (2 * stride));

                        #1;

                        if (curr_stage !== s) begin
                            $display(
                                "ERROR: stage mismatch; expected=%0d got=%0d",
                                s, curr_stage
                            );
                            errors = errors + 1;
                        end

                        if (idx_a !== exp_a[ADDR_WIDTH-1:0]) begin
                            $display(
                                "ERROR: idx_a expected=%0d got=%0d",
                                exp_a, idx_a
                            );
                            errors = errors + 1;
                        end

                        if (idx_b !== exp_b[ADDR_WIDTH-1:0]) begin
                            $display(
                                "ERROR: idx_b expected=%0d got=%0d",
                                exp_b, idx_b
                            );
                            errors = errors + 1;
                        end

                        if (k !== exp_k[ADDR_WIDTH-1:0]) begin
                            $display(
                                "ERROR: k expected=%0d got=%0d",
                                exp_k, k
                            );
                            errors = errors + 1;
                        end

                        last_in_stage =
                            (b == stride - 1) &&
                            (g == groups - 1);

                        // Advance one butterfly
                        tick;

                        if (last_in_stage) begin
                            if (done_stage !== 1'b1) begin
                                $display(
                                    "ERROR: done_stage not asserted at stage %0d end",
                                    s
                                );
                                errors = errors + 1;
                            end
                        end
                        else begin
                            if (done_stage !== 1'b0) begin
                                $display(
                                    "ERROR: done_stage asserted too early"
                                );
                                errors = errors + 1;
                            end
                        end

                        // Only the final butterfly of the
                        // final stage should complete the FFT.
                        if ((s == stage_count - 1) &&
                            last_in_stage) begin
                            if (done_fft !== 1'b1) begin
                                $display(
                                    "ERROR: done_fft not asserted at FFT completion"
                                );
                                errors = errors + 1;
                            end
                        end
                        else begin
                            if (done_fft !== 1'b0) begin
                                $display(
                                    "ERROR: done_fft asserted too early"
                                );
                                errors = errors + 1;
                            end
                        end

                    end
                end
            end

            if (done_fft === 1'b1)
                $display("PASS: AGU completed N=%0d", n_value);
            else begin
                $display("ERROR: FFT completion flag is not high");
                errors = errors + 1;
            end

        end
    endtask

    initial begin
        reset = 1'b0;
        start = 1'b0;
        next_step = 1'b0;
        N = 0;
        errors = 0;

        // Apply active-low reset
        #12;
        reset = 1'b1;

        // Supported power-of-two FFT sizes
        run_agu_case(4, 2);
        run_agu_case(8, 3);
        run_agu_case(16, 4);

        $display("");
        $display("------------------------------------");

        if (errors == 0)
            $display("ALL AGU TESTS PASSED");
        else
            $display("AGU TESTS FAILED: %0d errors", errors);

        $display("------------------------------------");
        $finish;
    end

endmodule
