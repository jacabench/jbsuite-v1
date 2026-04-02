# Contributing to the JACABench benchmark suite

Thank you for considering contributing to **jbsuite**. 

This document outlines the process for contributing to the repository.
Any intention to contribute shall be sent to: jacabenchsuite@gmail.com

## 🛠 Types of Contributions We Expect

We expect benchmarks that are ....

## 📁 Required Directory Structure

When preparing a new benchmark, please organize your files following our standard hierarchy to keep the repository clean and standardized.


## ⚖️ Licensing

By contributing to **jbsuite**, you agree that your contributions will be licensed under the License terms shown in the main folder. 

You also declare that you are the original author of the code you are submitting, or that you have the explicit rights to submit it under this open-source license. Please ensure that no proprietary, encrypted, or vendor-locked Intellectual Property (IP) is included in your Pull Request.


## 🚀 Contribution Process

### 1. Open an Issue First
The first step is to open an Issue proposing your new benchmark. Use that space to describe your proposal and explain why you think the benchmark is relevant to the community.   

### 2. Fork and Branch
Fork the repository to your own GitHub account and create a new branch for your feature:
`git checkout -b feature/add-your-benchmark`

### 3. Evaluating your code
The benchmark must be self-contained. All the libraries, testbenchs, datasets, etc, must be provided. 
At least a CPU and an HLS version must be ready to use. Could you synthesize the HLS design using your target toolchain and ensure it works (e.g., verify via co-simulation)?

### 4. Report Your Baselines (`metadata.json`)
Show the baseline results. Include this data in a `metadata.json` file:
* **Target Hardware:** Exact FPGA part number or MCU (e.g., Xilinx Artix-7 xc7a100t)
* **Toolchain:** Exact EDA tool and version (e.g., Vivado 2023.2)
* **Resource Utilization:** LUTs, FFs, DSP slices, and Block RAM, etc.
* **Performance:** Clock Frequency ($F_{max}$), Latency (in clock cycles), Throughput,  etc.

### 5. Submit the Pull Request (PR)
Open a PR against the `main` branch of the upstream repository. When you open the PR, our automated checklist template will appear in the text box—please fill it out completely. 

Make sure to provide a clear summary of your benchmark, instructions on how to run your scripts, and a link back to your original Issue using GitHub's closing keywords (e.g., type `Resolves #12` or `Closes #12` in your PR description).

### 6. Review Process
A maintainer will pull your branch, run your scripts on the reference toolchain to verify your claims, and review your code. Be prepared to answer questions or make minor formatting adjustments during the review!
