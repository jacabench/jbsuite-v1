# --- Vitis HLS Automation Script ---
set original_dir [pwd]
open_project -reset "cnn_3_layer_vitis_project"
set_top cnn
add_files "$original_dir/conv3.cpp"
add_files "$original_dir/conv3.h"
add_files -tb "$original_dir/conv_tb3.cpp"
open_solution -flow_target vitis "solution1"
set_part {xczu7ev-ffvc1156-2-e}
create_clock -period 10ns -name default
set sim_dir "cnn_3_layer_vitis_project/solution1/csim/build"
file mkdir $sim_dir
file copy -force "$original_dir//content/drive/MyDrive/MyBox/prof/projetos de pesquisa/pauNaJaca/dataset_x/WISDM/HAR-Dataset/test_windowed_channel3_windowsize10_class6.dat" $sim_dir
file copy -force "$original_dir/W1.bin" $sim_dir
file copy -force "$original_dir/B1.bin" $sim_dir
file copy -force "$original_dir/W2.bin" $sim_dir
file copy -force "$original_dir/B2.bin" $sim_dir
file copy -force "$original_dir/W3.bin" $sim_dir
file copy -force "$original_dir/B3.bin" $sim_dir
file copy -force "$original_dir/W_fc1.bin" $sim_dir
file copy -force "$original_dir/B_fc1.bin" $sim_dir
file copy -force "$original_dir/W_fc2.bin" $sim_dir
file copy -force "$original_dir/B_fc2.bin" $sim_dir
csim_design
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog
exit