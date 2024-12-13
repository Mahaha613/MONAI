#!/bin/bash

EXP_DIR_2=$(date +%m_%d)_BestArgs_sourceData_repair_iloc


nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=my_tr_trs \
--data_path=BSHD_src_data/image \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
--fig_save_name=${EXP_DIR_2}.png \
--device=0  > css/${EXP_DIR_2}.log 2>&1 &

