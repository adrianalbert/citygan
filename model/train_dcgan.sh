#!/bin/bash
python train_dcgan.py \
--dataset='spatial-maps' \
--dataroot=/home/data/world-cities/spatial-maps/splits/0/ \
--outf=/home/workspace/citygan/dcgan-0-res128-ngf128-ndf32-nz200 \
--imageSize=128 \
--nz=200 \
--cuda \
--nc=1 \
--lr=0.0002 \
--lr_halve=100 \
--ngf=128 \
--ndf=32 \
--niter=150 \
--gpu_ids='0,1,2,3' \
--workers=12 
# --custom_loader=True \
# --normalize=True \
# --rotate_angle=0 \
# --flips=False \
# --take_log=False \
# --use_channels="0,1,2" \
# --fix_dynamic_range=True \