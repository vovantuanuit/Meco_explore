#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume ./supernet-tiny.pth \
--min-param-limits 5 --param-limits 7 --data-set EVO_IMNET --path-save-api './AutoFM_CVPR2022_API_5_7M.json' --num-net 1000