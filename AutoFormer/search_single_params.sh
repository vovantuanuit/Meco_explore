#!/bin/bash
python evolution_get_params.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume /home/tuanvovan/MeCo/Meco_explore/Cream/AutoFormer/supernet-tiny.pth \
--min-param-limits 5 --param-limits 7 --data-set EVO_IMNET