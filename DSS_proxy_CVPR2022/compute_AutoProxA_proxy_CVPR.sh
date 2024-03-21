#!/bin/bash
python compute_AutoProxA_proxy.py --data-path '/home/xinda/Projects/Meco_explore_20240320/AutoFormer/dataset/imagenet' --gp \
 --change_qk --relative_position --dist-eval --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search' --api '/home/xinda/Projects/Meco_explore_20240320/DSS_proxy_CVPR2022/AutoFM_CVPR2022_API_5_7M.json' --save-result-json './json_AutoProxA_proxy_CVPR.json'


