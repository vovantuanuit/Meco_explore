#!/bin/bash
python evolution_meco.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
--change_qk --relative_position --dist-eval --cfg ./experiments/supernet/supernet-AAAI.yaml --resume ./supernet-tiny.pth \
--min-param-limits 1 --param-limits 12 --data-set EVO_IMNET --api './AutoFM_AAAI24_API.json' --zero-cost 'meco' --start 0 --end 2 --save-json './AutoFM_AAAI24_results.json'