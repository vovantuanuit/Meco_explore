# Explore Meco

## Installation

```
Python >= 3.6
PyTorch >= 2.0.0
nas-bench-201
```

## Preparation

1. Download three datasets (CIFAR-10, CIFAR-100, ImageNet16-120) from [Google Drive](https://drive.google.com/drive/folders/1T3UIyZXUhMmIuJLOBMIYKAsJknAtrrO4),  place them into the directory `./zero-cost-nas/_dataset`
2. Download the [`data` directory](https://drive.google.com/drive/folders/18Eia6YuTE5tn5Lis_43h30HYpnF9Ynqf?usp=sharing) and save it to the root folder of this repo. 
3. Download the benchmark files of NAS-Bench-201 from [Google Drive](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view) , put them into the directory `./data`
4. Download the [NAS-Bench-101 dataset](https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord), put it into the directory `./data`
5. Install `zero-cost-nas`
 ```bash
 cd zero-cost-nas
 pip install .
 cd ..
 ```

## Usage/Examples

### Correlation Experiment on NAS-Bench201

```bash
cd zero-cost-nas

python NAS_Bench_201.py
```




### Experiments on MobileNet OFA

1. Download ImageNet-1k Dataset:
Due to the ImageNet-1k is to large, and we use one sample to compute Meco. Therefor:
+ Download the Imagenet-100 from the links:
https://sutdapac-my.sharepoint.com/:u:/g/personal/vovan_tuan_sutd_edu_sg/EaWA3oLM575Nv_0mXoL7vlYBlhJ5IZvGc1YbjIjkavovUg?e=e5v6HM
and put the train folder in to ./zero-cost-nas/ZiCo/dataset/imagenet/

+ We need use full validation on ImageNet-1k, so we download Val folder of Imagenet1k by scripts, and put the val folder in to the ./zero-cost-nas/ZiCo/dataset/imagenet/
  ```bash

bash dowload_image1k.sh
```

1. Run Zero-Cost-PT with appointed zero-cost proxy:
```bash
cd exp_scripts
bash zerocostpt_nb201_pipline.sh --metric [metric] --batch_size [batch_size] --seed [seed]
```
You can choice metric from `['snip', 'fisher', 'synflow', 'grad_norm', 'grasp', 'jacob_cov','tenas', 'zico', 'meco'] `

### Experiments on DARTS-CNN Space

#### 1. DARTS CNN Space

```bash
cd exp_scripts
bash zerocostpt_darts_pipline.sh --metric [metric] --batch_size [batch_size] --seed [seed]
```

#### 2. DARTS Subspaces S1-S4

````bash
cd exp_scripts
bash zerocostpt_darts_pipline.sh --metric [metric] --batch_size [batch_size] --seed [seed] --space [s1-s4]
````

## Reference

Our code is based on [Zero-Cost-PT](https://github.com/zerocostptnas/zerocost_operation_score) and [Zero-Cost-NAS](https://github.com/SamsungLabs/zero-cost-nas).
