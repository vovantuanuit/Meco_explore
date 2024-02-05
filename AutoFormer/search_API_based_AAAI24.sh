#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=2 --use_env evolution.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-AAAI.yaml --resume ./supernet-tiny.pth \
--min-param-limits 1 --param-limits 12 --data-set EVO_IMNET --path-save-api './AutoFM_AAAI24_API.json' --num-net 500