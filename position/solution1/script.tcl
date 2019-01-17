############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
open_project position
set_top position
add_files lstm_hls.h
add_files lstm_sensor.cpp
add_files params.h
add_files -tb main.cpp
open_solution "solution1"
set_part {xcvu9p-flgb2104-2-i} -tool vivado
create_clock -period 10 -name default
#source "./position/solution1/directives.tcl"
csim_design -clean -compiler gcc -setup
csynth_design
cosim_design
export_design -format ip_catalog
