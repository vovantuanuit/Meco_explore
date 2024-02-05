#!/bin/bash
python evolution_meco.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-T.yaml --resume /home/tuanvovan/MeCo/Meco_explore/Cream/AutoFormer/supernet-tiny.pth \
--min-param-limits 5 --param-limits 7 --data-set EVO_IMNET --api './AutoFM_CVPR2022_API_5_7M.json' --zero-cost 'meco' --start 0 --end 125 --save-json './AutoFM_CVPR_results.json'