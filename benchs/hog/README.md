# HOG Pedestrian Detector

## Overview
This benchmark implements the Histogram of Oriented Gradients (HOG) algorithm in C, specifically targeting pedestrian detection on RGB images. The detector operates as a multi-stage pipeline consisting of three major computational levels: image pyramid generation to support multi-scale detection, sliding-window scanning across the resized images, and the core feature descriptor computation. 

By capturing the distribution of local intensity gradients and edge directions, the HOG descriptor models object shape and appearance. This repository provides two distinct architectural versions of the algorithm—Dynamic (hog-ptr) and Static (hog-array)—to accommodate different execution environments, ranging from general-purpose CPUs to specialized hardware accelerators.

---

### To run the detector, follow the steps:

1. Open a new terminal

2. Navigate to the \<hog-version\> folder

3. Compile using *make*

4. Run the detector from the \<hog-version\>/bin folder
