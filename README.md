# jbsuite-v1
Version 1 of the JACA Benchmark Suite

HAR-CNN:
- Download this repo and unzip de folder dataset.zip to create a folder dataset with train and test data
[To train the model on your CPU (x86)]
1.Open a new terminal
2.Navigate to the x86/backprop folder
3.Compile the trainer: make
4.Run the trainer in the x86/bin folder. This creates the .bin weight files and the normalization .txt file needed for both x86 and HLS inference

[To run inference on your CPU (x86)]
Note that: Only float versions are prepared to run on a CPU
1.Open a new terminal
2.Navigate to the x86/scr folder
3.Compile the model: make
4.Run inference from the x86/bin folder.

[To synthesize the hardware with Vitis HLS]
1.Navigate to the VitisHLS/src folder
2.Run the script: vitis_hls -f run_hls.tcl.
3.It may take long time since it does simulation, generated RTL and then do co-simulation). You can avoid going through some steps by commenting the respective lines in the run_hls.tcl script.
