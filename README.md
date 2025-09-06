# Light4Mars：A Lightweight Transformer Model for Semantic Segmentation on Unstructured Environment Like Mars

## Introduction
This repository is the code implementation of the paper Light4Mars：A Lightweight Transformer Model for Semantic Segmentation on Unstructured Environment Like Mars, which is based on the [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation) project.

## Installation

Step 0: Create a virtual environment named light4mars and activate it.
```
conda create -n light4mars python=3.8 -y
conda activate light4mars
```

Step 1: Install PyTorch 2.0.1 and torchvision 0.15.2.
```
pip install torch==2.0.1
pip install torchvision==0.15.2
```

Step 2: Install MMCV and mmsegmentation.
```
pip install -U openmim
mim install mmengine==0.8.4
mim install mmcv=2.0.0
pip install mmsegmentation=1.1.1
```

## Dataset Preparation
The dataset used in the paper is [**SynMars-TW**](https://github.com/CVIR-Lab/SynMars/tree/SynMars-TW), which is subset of the open source unstructured environmental fine-grained synthetic dataset [**SynMars**](https://github.com/CVIR-Lab/SynMars) based on real data from the TianWen-1 mission. Please download the [**SynMars-TW**](https://github.com/CVIR-Lab/SynMars/tree/SynMars-TW) dataset and set it according to the [**MMSegmentation**](https://github.com/open-mmlab/mmsegmentation) data format.
### Available datasets
| Name | Size(resolution) | Object |  Support | View angle | Bsed Mission |
| :--------- | :---------:  | ---------: | ---------:| ---------: |--------- |
| MarsData    | 8390 (512*512)      | Rock  | semantic segmentation  |Rover  | Curiosity rover  |
| Marsscape    | 195(Panorama,3779 subimages )      |  All terrain | semantic segmentation  | Rover   | Curiosity rover  |
| SynMars    | 60,000(1024*1024)     | Rock   | semantic segmentation| Rover  | TianWen-1  |
| SynMars-TW   | 21,000(512*512)     | All terrain  | semantic segmentation  | Rover | TianWen-1  |
| SynMars-Air   | 11,700(512*512)  | All terrain  | semantic segmentation  | Air  | TianWen-1  |
## Model Training
```
python train.py configs/light4mars/light4mars-b_synmars-tw.py
```
## Model Testing
```
python test.py configs/light4mars/light4mars-b_synmars-tw.py
```
## Citation

If you use the code or performance benchmarks of this project in your research, please refer to the following bibtex citation of Light4Mars.
```
@article{xiong2024light4mars,
  title={Light4Mars: A lightweight transformer model for semantic segmentation on unstructured environment like Mars},
  author={Xiong, Yonggang and Xiao, Xueming and Yao, Meibao and Cui, Hutao and Fu, Yuegang},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={214},
  pages={167--178},
  year={2024},
  publisher={Elsevier}
}
```

