# jbsuite-v1
Version 1 of the JACA Benchmark Suite

### HAR-CNN: download this repo and unzip de file dataset.zip in its current location to create a folder named dataset with train and test data.

#### To train the model on your CPU (x86)
1. Open a new terminal

2. Navigate to the x86/backprop folder

3. Compile the trainer: make

4. Run the trainer in the x86/bin folder. This creates the .bin weight files and the normalization .txt file needed for both x86 and HLS inference

#### To run inference on your CPU (x86)
Note that only float versions are prepared to run on a CPU
1. Open a new terminal

2. Navigate to the x86/scr folder

3. Compile the model: make

4. Run inference from the x86/bin folder

#### To synthesize the hardware with Vitis HLS
For each HAR-CNN model architecture configuration, there are two versions named internalWeights and externalWeights. The former has the trained model weights hardcoded, while the latter reads the weights from a testbench source.
1. Navigate to the VitisHLS folder and choose internalWeights or externalWeights

2. Run the script: vitis_hls -f run_hls.tcl

3. It may take a very long time to conclude, since it does simulation, generates RTL and then does co-simulation. You can avoid going through some steps by commenting the respective lines in the run_hls.tcl script (by default, only simulation is enabled).

#### HLS Simulation on device 'xczu7ev-ffvc1156-2-e' running inference on 316462 test samples:
Model version: fix_CNN_1Conv_2FC_1D 
Correctly Classified: 233861 / 316462
Accuracy: 73.90%
CSim: Elapsed time: 00:37:31

Model version: fix_CNN_3Conv_2FC_1D 
Correctly Classified: 227935 / 316462
Accuracy: 72.03%
CSim: Elapsed time: 01:10:35

Model version: fix_CNN_5Conv_1FC_1D 
Correctly Classified: 232476 / 316462
Accuracy: 73.46%
CSim: Elapsed time: 03:33:37


