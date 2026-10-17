`timescale 1ns/1ps

module tb_multiplier;

    // ========================================
    // FP4 SCALAR MULTIPLIER
    // ========================================
    reg  [3:0] fp4_a, fp4_b;
    wire [3:0] fp4_y;

    fp4_mul dut_fp4_mul (
        .a(fp4_a),
        .b(fp4_b),
        .out(fp4_y)
    );

    // ========================================
    // FP8 SCALAR MULTIPLIER
    // ========================================
    reg  [7:0] fp8_a, fp8_b;
    wire [7:0] fp8_y;

    fp8_mul dut_fp8_mul (
        .a(fp8_a),
        .b(fp8_b),
        .out(fp8_y)
    );

    // ========================================
    // FP4 COMPLEX MULTIPLIER
    // ========================================
    reg [3:0] fp4_ca, fp4_cb, fp4_cc, fp4_cd;
    wire [3:0] fp4_c_real, fp4_c_imag;

    fp4_cmul dut_fp4_cmul (
        .a(fp4_ca),
        .b(fp4_cb),
        .c(fp4_cc),
        .d(fp4_cd),
        .out_real(fp4_c_real),
        .out_imag(fp4_c_imag)
    );

    // ========================================
    // FP8 COMPLEX MULTIPLIER
    // ========================================
    reg [7:0] fp8_ca, fp8_cb, fp8_cc, fp8_cd;
    wire [7:0] fp8_c_real, fp8_c_imag;

    fp8_cmul dut_fp8_cmul (
        .a(fp8_ca),
        .b(fp8_cb),
        .c(fp8_cc),
        .d(fp8_cd),
        .out_real(fp8_c_real),
        .out_imag(fp8_c_imag)
    );

    integer errors;
    integer i, j;
    reg [3:0] saved_fp4_result;
    reg [7:0] saved_fp8_result;

    // ========================================
    // FP4 TEST TASK
    // ========================================
    task check_fp4_mul;
        input [3:0] lhs;
        input [3:0] rhs;
        input [3:0] expected_value;

        begin
            fp4_a = lhs;
            fp4_b = rhs;
            #2;

            if (fp4_y !== expected_value) begin
                $display(
                    "FP4 FAIL: %h * %h, expected=%h, got=%h",
                    lhs, rhs, expected_value, fp4_y
                );
                errors = errors + 1;
            end
            else begin
                $display(
                    "FP4 PASS: %h * %h = %h",
                    lhs, rhs, fp4_y
                );
            end
        end
    endtask

    // ========================================
    // FP8 TEST TASK
    // ========================================
    task check_fp8_mul;
        input [7:0] lhs;
        input [7:0] rhs;
        input [7:0] expected_value;

        begin
            fp8_a = lhs;
            fp8_b = rhs;
            #2;

            if (fp8_y !== expected_value) begin
                $display(
                    "FP8 FAIL: %h * %h, expected=%h, got=%h",
                    lhs, rhs, expected_value, fp8_y
                );
                errors = errors + 1;
            end
            else begin
                $display(
                    "FP8 PASS: %h * %h = %h",
                    lhs, rhs, fp8_y
                );
            end
        end
    endtask

    // ========================================
    // MAIN TEST SEQUENCE
    // ========================================
    initial begin
        errors = 0;

        fp4_a = 0;
        fp4_b = 0;
        fp8_a = 0;
        fp8_b = 0;

        fp4_ca = 0;
        fp4_cb = 0;
        fp4_cc = 0;
        fp4_cd = 0;

        fp8_ca = 0;
        fp8_cb = 0;
        fp8_cc = 0;
        fp8_cd = 0;

        // ------------------------------------
        // FP4 E2M1 MULTIPLICATION TESTS
        // 0x2 represents +1.0
        // 0x3 represents +1.5
        // ------------------------------------
        $display("");
        $display("========== FP4 MULTIPLIER ==========");

        check_fp4_mul(4'h0, 4'h3, 4'h0);
        check_fp4_mul(4'h2, 4'h2, 4'h2);
        check_fp4_mul(4'h3, 4'h2, 4'h3);
        check_fp4_mul(4'h3, 4'h3, 4'h4);
        check_fp4_mul(4'h4, 4'h2, 4'h4);
        check_fp4_mul(4'h6, 4'h2, 4'h6);
        check_fp4_mul(4'hA, 4'h2, 4'hA);
        check_fp4_mul(4'hB, 4'h2, 4'hB);

        // Exhaustive FP4 commutativity test
        $display("");
        $display("Checking FP4 commutativity...");

        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin

                fp4_a = i[3:0];
                fp4_b = j[3:0];
                #1;

                saved_fp4_result = fp4_y;

                fp4_a = j[3:0];
                fp4_b = i[3:0];
                #1;

                if (fp4_y !== saved_fp4_result) begin
                    $display(
                        "FP4 COMMUTATIVITY FAIL: %h, %h",
                        i[3:0], j[3:0]
                    );
                    errors = errors + 1;
                end
            end
        end

        // ------------------------------------
        // FP8 E4M3 MULTIPLICATION TESTS
        // 0x38 represents +1.0
        // 0x3C represents +1.5
        // ------------------------------------
        $display("");
        $display("========== FP8 MULTIPLIER ==========");

        check_fp8_mul(8'h00, 8'h3C, 8'h00);
        check_fp8_mul(8'h38, 8'h38, 8'h38);
        check_fp8_mul(8'h3C, 8'h38, 8'h3C);
        check_fp8_mul(8'h3C, 8'h3C, 8'h41);
        check_fp8_mul(8'h40, 8'h40, 8'h48);
        check_fp8_mul(8'hB8, 8'h38, 8'hB8);

        // FP8 commutativity test
        fp8_a = 8'h3C;
        fp8_b = 8'h40;
        #2;

        saved_fp8_result = fp8_y;

        fp8_a = 8'h40;
        fp8_b = 8'h3C;
        #2;

        if (fp8_y !== saved_fp8_result) begin
            $display("FP8 COMMUTATIVITY FAIL");
            errors = errors + 1;
        end
        else begin
            $display("FP8 COMMUTATIVITY PASS");
        end

        // ------------------------------------
        // FP4 COMPLEX MULTIPLICATION
        // (1 + j0) * (1 + j0) = 1 + j0
        // ------------------------------------
        $display("");
        $display("========== FP4 COMPLEX MULTIPLIER ==========");

        fp4_ca = 4'h2;
        fp4_cb = 4'h0;
        fp4_cc = 4'h2;
        fp4_cd = 4'h0;
        #2;

        if (fp4_c_real !== 4'h2 ||
            fp4_c_imag !== 4'h0) begin

            $display(
                "FP4 COMPLEX FAIL: expected real=2 imag=0, got real=%h imag=%h",
                fp4_c_real, fp4_c_imag
            );
            errors = errors + 1;
        end
        else begin
            $display("FP4 COMPLEX PASS: (1+j0)*(1+j0)");
        end

        // (1 + j1) * (1 + j1) = 0 + j2
        fp4_ca = 4'h2;
        fp4_cb = 4'h2;
        fp4_cc = 4'h2;
        fp4_cd = 4'h2;
        #2;

        if (fp4_c_real !== 4'h0 ||
            fp4_c_imag !== 4'h4) begin

            $display(
                "FP4 COMPLEX FAIL: expected real=0 imag=4, got real=%h imag=%h",
                fp4_c_real, fp4_c_imag
            );
            errors = errors + 1;
        end
        else begin
            $display("FP4 COMPLEX PASS: (1+j1)*(1+j1)");
        end

        // ------------------------------------
        // FP8 COMPLEX MULTIPLICATION
        // (1 + j0) * (1 + j0) = 1 + j0
        // ------------------------------------
        $display("");
        $display("========== FP8 COMPLEX MULTIPLIER ==========");

        fp8_ca = 8'h38;
        fp8_cb = 8'h00;
        fp8_cc = 8'h38;
        fp8_cd = 8'h00;
        #2;

        if (fp8_c_real !== 8'h38 ||
            fp8_c_imag !== 8'h00) begin

            $display(
                "FP8 COMPLEX FAIL: expected real=38 imag=00, got real=%h imag=%h",
                fp8_c_real, fp8_c_imag
            );
            errors = errors + 1;
        end
        else begin
            $display("FP8 COMPLEX PASS: (1+j0)*(1+j0)");
        end

        // (1 + j1) * (1 + j1) = 0 + j2
        fp8_ca = 8'h38;
        fp8_cb = 8'h38;
        fp8_cc = 8'h38;
        fp8_cd = 8'h38;
        #2;

        if (fp8_c_real !== 8'h00 ||
            fp8_c_imag !== 8'h40) begin

            $display(
                "FP8 COMPLEX FAIL: expected real=00 imag=40, got real=%h imag=%h",
                fp8_c_real, fp8_c_imag
            );
            errors = errors + 1;
        end
        else begin
            $display("FP8 COMPLEX PASS: (1+j1)*(1+j1)");
        end

        // ------------------------------------
        // FINAL TEST REPORT
        // ------------------------------------
        $display("");
        $display("====================================");

        if (errors == 0)
            $display("ALL MULTIPLIER TESTS PASSED");
        else
            $display("MULTIPLIER TESTS FAILED: %0d errors", errors);

        $display("====================================");

        $finish;
    end

endmodule
