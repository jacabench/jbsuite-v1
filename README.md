# jbsuite-v1
Version 1 of the JACA Benchmark Suite

## The benchmark suite is organized as follows:
jbsuite-v1/\
&nbsp;&nbsp;├── benchs\
&nbsp;&nbsp;│   ├── smooth\
&nbsp;&nbsp;│   ├── har-knn\
&nbsp;&nbsp;│   ├── hog\
&nbsp;&nbsp;│   └── har-cnn\
&nbsp;&nbsp;├── includes\
&nbsp;&nbsp;└── tools-libs\
&nbsp;&nbsp;└── bmplib

## JACABench HOME
To run the makefiles, you need to set the JBSPATH environment variable to point to the location of the JACABenchSuite directory.\ 
For example:\
Linux: export JBSPATH=.../JACABenchSuite\
Windows: set  JBSPATH=...\JACABenchSuite

## Datasets
Folders with the files with data from WIDSDM and PAMA2 datasets are available at: https://drive.google.com/drive/folders/1jp1rNl9nW6CpP6d0VK8ks2yT_KHXdkLT?usp=drive_link \
ZIP file: https://drive.google.com/file/d/1TAthMikrqrAEMU7JRsTiNZgU9zaQddDH/view?usp=sharing \
Those folders must be copied to the har-knn folder.

## Contact
Email: jacabenchsuite@gmail.com

## Citation
If you find the JACABench useful, please use the following citation:
```
@inproceedings{jacabenchARC2026,
  author    = {Jos{\'e} A.M. de Holanda and Vanderlei Bonato and Jo{\~a}o M.P. Cardoso and Jos{\'e} Mendes},
  title     = {{JACABench} -- The {JACA} Benchmark Suite for Embedded Computing},
  booktitle = {Applied Reconfigurable Computing. ARC 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```
