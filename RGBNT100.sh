#!/bin/bash
source activate base
cd /13994058190/WYH/MM_CLIP
pip install timm
pip install ftfy
python train_net.py --config_file /13994058190/WYH/MM_CLIP/configs/RGBNT100/TOP-ReID.yml
