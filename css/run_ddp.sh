#!/bin/bash

EXP_DIR_1=$(date +%m_%d)_BestArgs_repair_iloc_convMerging_DDP_BS4


nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--merging_type=conv \
--eval_num=24 \
--batch_size=4 \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_1} \
--fig_save_name=${EXP_DIR_1}.png \
--device=0,2  > css/${EXP_DIR_1}.log 2>&1 &

