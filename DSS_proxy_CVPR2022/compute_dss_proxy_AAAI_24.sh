#!/bin/bash
python compute_dss_proxy.py --data-path '/home/tuanvovan/MeCo/zero-cost-nas/ZiCo/dataset/imagenet' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search' --api path_to_AAAI24_api --save-result-json './json_dss_proxy_AAAI24.json'


