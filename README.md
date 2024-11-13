<div align=center>
  
# Perception- and Fidelity-aware Reduced-Reference Super-Resolution Image Quality Assessment

[[paper]](https://ieeexplore.ieee.org/document/10742110) [[code](https://github.com/xinyouu/PFIQA)]

</div>

## Overview
<p align="center"> <img src="overview.png" width="100%"> </p>

## Getting Started

### Prerequisites
- Linux
- NVIDIA GPU + CUDA CuDNN
- Python 3.7

### Dependencies
All dependencies for defining the environment are provided in `requirements.txt`.

### Dataset
- We conduct experiments on three widely-used SR-IQA benchmarks, including [WIND](https://ivc.uwaterloo.ca/database/WIND.html), [QADS](http://www.vista.ac.cn/super-resolution/) and [RealSRQ](https://github.com/Zhentao-Liu/RealSRQ-KLTSRQA).
- We split the dataset into training and testing sets with a ratio of 8:2. The training and testing lists are stored in train.txt and test.txt respectively, following the format: “SRimage_name#MOS#LRimage_name”. The dataloader (`data/RealSRQ.py`) can be modified to accommodate different txt file formats.


### Instruction
use `sh train.sh` or `sh test.sh` to train or test the model. You can also change the options in the `options/` as you like.

## Acknowledgment
The codes are based on [AHIQ](https://github.com/IIGROUP/AHIQ). Thanks for their awesome works.

## Citation
```bibtex
@article{lin2024perception,
  title={Perception-and Fidelity-aware Reduced-Reference Super-Resolution Image Quality Assessment},
  author={Lin, Xinying and Liu, Xuyang and Yang, Hong and He, Xiaohai and Chen, Honggang},
  journal={IEEE Transactions on Broadcasting},
  year={2024},
  publisher={IEEE}
}
```

## Contact
For any question about our paper or code, please emial `linxinying@stu.scu.edu.cn`.


