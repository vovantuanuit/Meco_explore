# Explore Meco

## Installation

```
conda env create -f environment.yml
conda activate meco
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

python NAS_Bench_201.py --dataset 'cifar10/cifar100/ImageNet16-120'
```
can get the json result saved at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/EkEO5RQtDt5PrsUpbxzldJ8BXlhEiOeNVSiTUCj9er-nFw?e=MGHniT


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
2. Get API net and test-accuracy MobileNet OFA on ImageNet1k:
set path dataset and path save json in mobilenet_OFA_eval_correlation_imagnet_1k_for_API.py as:
```bash
path_data_imagenet = '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet'

path_save_json_api = "./MobileNet_OFA_1000arc_api.json"
````
```bash
cd zero-cost-nas
python mobilenet_OFA_eval_correlation_imagnet_1k_for_API.py
```


3. Compute correlation Meco score and Test-Accuracy and save json Meco value for MobileNet OFA on ImageNet1k:
can get API with net setting and test-accuracy at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/EjromH0Fu2BNvbE8Zc5OfvYBNsOGDQ2RjIKa3QLTuTsyfQ?e=O7ZDci
set path dataset, path api and path save json in mobilenet_OFA_eval_correlation_imagenet_1k.py as:
```bash
path_data_imagenet = '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet'
net_api_save_path= './MobileNet_OFA_1000arc_api_42_3.json'
path_save_json = "/home/tuanvovan/MeCo/zero-cost-nas/save_data.json"
````

```bash
cd zero-cost-nas
###mece base
python mobilenet_OFA_eval_correlation_imagnet_1k_load_computemoce_base.py
###mece opt
python mobilenet_OFA_eval_correlation_imagnet_1k_load_computemoce_opt.py
###mece revised
python mobilenet_OFA_eval_correlation_imagnet_1k_load_computemoce_revised.py
```
can get the result json saved at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/EgwTYBEsXF9DhHmvdSp0_GMB7ws4P41dgblUS8oGDLHKIg?e=p33WcT

4. Save json Meco value for accross architecture design (MobileNetv2,Resnet18,Efficientnet-B0,ViT,MaxVit) on ImageNet1k:
set path dataset and path save json in accross_specific_architecture_torch.py as:
```bash
path_data_imagenet = '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet'
path_save_json = "/home/tuanvovan/MeCo/zero-cost-nas/save_data.json"
````
```bash
cd zero-cost-nas
python accross_specific_architecture_torch.py
```
can get the result json saved at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/EkEO5RQtDt5PrsUpbxzldJ8BXlhEiOeNVSiTUCj9er-nFw?e=MGHniT
### Experiments on AutoFM

Dowloading the pretrained Autoformer Supernet at here: https://drive.google.com/drive/folders/1HqzY3afqQUMI6pJ5_BgR2RquJU_b_3eg

1. Following CVPR22 paper:

Get architecture and test-accuracy save as API:

Change your Imanget1k path with --data-path 
```bash
cd AutoFormer
bash search_API_based_CVPR2022.sh
or 
python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume ./supernet-tiny.pth \
--min-param-limits 5 --param-limits 7 --data-set EVO_IMNET --path-save-api './AutoFM_CVPR2022_API_5_7M.json' --num-net 1000
```
The API with archtecture and accuracy saved at --path-save-api './AutoFM_CVPR2022_API_5_7M.json' --num-net 1000
can get API at here: https://sutdapac-my.sharepoint.com/:u:/g/personal/vovan_tuan_sutd_edu_sg/Ee95Cf3zmt9KmXngIpVma4ABaT6f8CuZBcB_ZcqURHfftg?e=07nzNz

Compute meco, meco_opt, meco_revised based oh API have saved: './AutoFM_CVPR2022_API_5_7M.json'

Change 'meco', 'meco_opt' or 'meco_revise' for --zero-cost .

Noted: For the compute meco, we can't compute meco for 1000 in one process (will crash memory). Therefore we need split 125 architecture for each process, change --start and --end (0,125), (125,250), ... , (875,1000)
```bash
cd AutoFormer
bash compute_meco_CVPR.sh
or 
python evolution_meco.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume /home/tuanvovan/MeCo/Meco_explore/Cream/AutoFormer/supernet-tiny.pth \
--min-param-limits 5 --param-limits 7 --data-set EVO_IMNET --api './AutoFM_CVPR2022_API_5_7M.json' --zero-cost 'meco' --start 0 --end 125 --save-json './AutoFM_CVPR_results.json'
```

The result json will saved at './AutoFM_CVPR_results.json'

The json result when compute proxy at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/EiVabX6uXaFCrBMUSjsZquIBWyGkNlqJLtVgFt-gJcEGcg?e=riyU1c

2.Following AAAI24 paper:

Get architecture and test-accuracy save as API:
Change your Imanget1k path with --data-path 
```bash
cd AutoFormer
bash search_API_based_AAAI24.sh
or 
python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-AAAI.yaml --resume ./supernet-tiny.pth \
--min-param-limits 1 --param-limits 12 --data-set EVO_IMNET --path-save-api './AutoFM_AAAI24_API.json' --num-net 500
```
The API with archtecture and accuracy saved at --path-save-api './AutoFM_AAAI24_API.json' --num-net 500

Can get the AutoFM_AAAI24_API at here: https://sutdapac-my.sharepoint.com/:u:/g/personal/vovan_tuan_sutd_edu_sg/EeqiPrBH6TxCknkWIneMjh4BJunkw0sgJkRrFC_jvQIdIQ?e=24qfyA

Compute meco, meco_opt, meco_revised based oh API have saved: './AutoFM_AAAI24_API.json'

change 'meco', 'meco_opt' or 'meco_revise' for --zero-cost . 

Noted: For the compute meco, we can't compute meco for 500 in one process (will crash memory). Therefore we need split 125 architecture for each process, change --start and --end (0,125), (125,250), ... , (375,500)
```bash
cd AutoFormer
bash compute_meco_AAAI.sh
or 
python evolution_meco.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-AAAI.yaml --resume ./supernet-tiny.pth \
--min-param-limits 1 --param-limits 12 --data-set EVO_IMNET --api './AutoFM_AAAI24_API.json' --zero-cost 'meco' --start 0 --end 125 --save-json './AutoFM_AAAI24_results.json'
```
The result json will saved at './AutoFM_AAAI24_results.json'

Can get the computed results at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/ElsY5s1M4-RKvo_BuWDj2TABThJgT1Plb7Xj2da5lW1hug?e=HiguYS

### Compute DSS proxy:
1. Compute dss proxy for VIT API benchmark have save base on CVPR: 
AutoFM_CVPR2022_API_5_7M.json

change path api json for --api
```bash
cd DSS_proxy_CVPR2022
bash compute_dss_proxy_CVPR.sh
or 
python compute_dss_proxy.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search' --api path_to_CVPR_api --save-result-json './json_dss_proxy_CVPR.json'
```
The compted dss proxy result json can get at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/EicFu3XR8U9Asg0Ytr5Zg5YBXR-1H7qG86riv7if8YV2_w?e=bkb52j

**Compute Auto-A proxy in AAAI-24 paper on AutoFormer CVPR search space:**
hange path api json for --api
```bash
cd DSS_proxy_CVPR2022
bash compute_AutoProxA_proxy_CVPR.sh.sh
or 
#!/bin/bash
python compute_AutoProxA_proxy.py --data-path '/home/xinda/Projects/Meco_explore_20240320/AutoFormer/dataset/imagenet' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search' --api '/home/xinda/Projects/Meco_explore_20240320/DSS_proxy_CVPR2022/AutoFM_CVPR2022_API_5_7M.json' --save-result-json './json_AutoProxA_proxy_CVPR.json'
```

2. Compute dss proxy for VIT API benchmark have save base on AAAI: 
AutoFM_AAAI24_API.json

change path api json for --api
```bash
cd DSS_proxy_CVPR2022
bash compute_dss_proxy_AAAI_24.sh
or 
python compute_dss_proxy.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search' --api path_to_AAAI24_api --save-result-json './json_dss_proxy_AAAI24.json'
```
The compted dss proxy result json can get at here: https://sutdapac-my.sharepoint.com/:f:/g/personal/vovan_tuan_sutd_edu_sg/Eodd5-wrC3hDo1IBwSF2vmABegXzwhkaoyZz7zXcxEuDkg?e=le5Evh

## Reference

Our code is based on [MeCO](https://github.com/HamsterMimi/MeCo).
