## apps.csv 

This file contains the application you have data from that you wish to plot. The file is structured as:

name, AI, FLOPS, 0 

The fourth column is not used at the moment. 

## hw.csv 

This file contains the peak performance and bandwidth for the architectures you are interested in doing the roofline analysis. The file is structured as

name, peak performance, peak bandwidth 

## How to run 

Peak bandwidth must be specified in GB/s
Peak performance must be specified in GFLOP/s
Arithemtic intensity is specified in FLOP/byte
Performance is specified in GFLOP/s

python3 roofline.py -i hw.csv -a apps.csv
