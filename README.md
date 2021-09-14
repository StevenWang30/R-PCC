# R-PCC

 A baseline for range image-based large-scale point cloud compression based on region segmentation and point-plane mixing modeling.
The basic compressor is used to compress the info data and residual data.  

## Overview
- [Installation](#installation)
- [Getting Started](#get-started)

## Installation
1. Install CUDA and C++ extensions. CUDA is used in FPS implementation, and some modules are implemented in C++ using pybind11.  

   ```
   python setup.py develop
   ```

2. Uninstall (if need): 
   ```
   python3 setup.py develop --uninstall --user 
   ```
## Usage

### Config Files
#### compressor.yaml
 - BASIC_COMPRESSOR_NAME: 'bzip2'
   
   The name of the basic compressor. Choose in 'lz4', 'bzip2', and 'deflate'.
      - Compression rate: bzip2 > deflate > lz4
      - Compression speed:  lz4 : deflate : bzip2 = 300 : 100 : 1

 - CLUSTER_NUM: 100
   
   The number of clusters after FPS.

 - ACCURACY: 0.02

   The maximum reconstructed error. For example, 0.02 means the difference 
   between the reconstructed range image and the original range image must smaller than 0.02m.
   
### Compress

```
python tools/compress.py --input assets/example_data/example.bin --output example.rpcc --dataset KITTI
```

### Decompress
```
python tools/decompress.py --input example.rpcc --output reconstructed.pcd --dataset KITTI
```
