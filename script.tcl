open_project "hls_lstm"
set_top position
add_files src/lstm_sensor.cpp -cflags "-Wno-unknown-pragmas -I weights"
add_files -tb testbench/main.cpp -cflags "-Wno-unknown-pragmas -I src/ -I weights" 
open_solution -reset "origin"
set_part {xc7z020clg400-1}
create_clock -period 200MHz -name default
config_schedule -effort medium -relax_ii_for_timing
config_interface -m_axi_addr64
csim_design
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog
