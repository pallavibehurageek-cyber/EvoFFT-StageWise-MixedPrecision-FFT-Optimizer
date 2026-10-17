# vivado_synthesis.tcl
# Now extracts critical path delay for latency objective

set design_name  [lindex $argv 0]
set csv_output   [lindex $argv 1]
set clock_period [lindex $argv 2]
set core_file    [lindex $argv 3]
set top_file     [lindex $argv 4]
set verilog_dir  [lindex $argv 5]

set top_module "${design_name}_top"

puts "INFO: design_name  = $design_name"
puts "INFO: top_module   = $top_module"
puts "INFO: clock_period = $clock_period ns"

# Create in-memory project
create_project -in_memory -part xc7z020clg484-1

# Add sources
foreach f [glob ${verilog_dir}/*.v] {
    add_files -norecurse $f
}
add_files -norecurse $core_file
add_files -norecurse $top_file

set_property include_dirs $verilog_dir [current_fileset]
set_property top $top_module [current_fileset]
update_compile_order -fileset sources_1

# Synthesis
synth_design -top $top_module -part xc7z020clg484-1 -mode out_of_context
create_clock -period $clock_period -name clk [get_ports clk]
opt_design

# Area (LUTs)
set util_rpt_file "/tmp/${design_name}_util.rpt"
report_utilization -file $util_rpt_file

set lut_count 0
if {[file exists $util_rpt_file]} {
    set fp [open $util_rpt_file r]
    set content [read $fp]
    close $fp
    foreach line [split $content "\n"] {
        if {[regexp {^\|\s*(CLB LUTs|Slice LUTs)\s*\|\s*(\d+)\s*\|} $line -> _lbl val]} {
            set lut_count [string trim $val]
            break
        }
    }
    if {$lut_count == 0} {
        set lut_logic 0; set lut_mem 0
        foreach line [split $content "\n"] {
            if {[regexp {^\|\s*LUT as Logic\s*\|\s*(\d+)\s*\|} $line -> val]} { set lut_logic [string trim $val] }
            if {[regexp {^\|\s*LUT as Memory\s*\|\s*(\d+)\s*\|} $line -> val]} { set lut_mem [string trim $val] }
        }
        set lut_count [expr {$lut_logic + $lut_mem}]
    }
}

# Power
set power_rpt_file "/tmp/${design_name}_power.rpt"
report_power -file $power_rpt_file
set total_power 0.0
if {[file exists $power_rpt_file]} {
    set fp [open $power_rpt_file r]
    set content [read $fp]
    close $fp
    foreach line [split $content "\n"] {
        if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} $line -> val]} {
            set total_power [string trim $val]
            break
        }
    }
}

# Timing - Critical Path Delay (on-chip latency source)
set timing_rpt_file "/tmp/${design_name}_timing.rpt"
report_timing_summary -file $timing_rpt_file -delay_type max -max_paths 10

set wns "N/A"
set critical_path_delay 100.0   ;# default heavy penalty

if {[file exists $timing_rpt_file]} {
    set fp [open $timing_rpt_file r]
    set content [read $fp]
    close $fp

    if {[regexp {WNS\(ns\)[^\n]*\n[^\n]*\n\s*([-0-9.]+)} $content -> val]} {
        set wns [string trim $val]
    }

    # Extract Data Path Delay
    if {[regexp {Data Path Delay:\s+([0-9.]+)ns} $content -> path_delay]} {
        set critical_path_delay [string trim $path_delay]
    } elseif {$wns != "N/A"} {
        set wns_val [expr {double($wns)}]
        set critical_path_delay [expr {$clock_period - $wns_val}]
    }
}

# Write CSV with critical path delay (on-chip latency)
set csv_dir [file dirname $csv_output]
file mkdir $csv_dir

set fp [open $csv_output w]
puts $fp "Metric,Value"
puts $fp "design_name,$design_name"
puts $fp "top_module,$top_module"
puts $fp "lut_count,$lut_count"
puts $fp "total_power_w,$total_power"
puts $fp "wns_ns,$wns"
puts $fp "critical_path_delay_ns,$critical_path_delay"
puts $fp "clock_period_ns,$clock_period"
close $fp

puts stdout "INFO: Synthesis complete  : $design_name"
puts stdout "INFO:   LUTs             = $lut_count"
puts stdout "INFO:   Power            = $total_power W"
puts stdout "INFO:   WNS              = $wns ns"
puts stdout "INFO:   Crit Path Delay  = $critical_path_delay ns"
puts stdout "INFO:   CSV              = $csv_output"
