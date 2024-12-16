#!/bin/bash
# eta_min=1e-5
EXP_DIR_1=$(date +%m_%d)_BestArgs_repairIloc_convMerging_6
EXP_DIR_2=$(date +%m_%d)_BestArgs_repairIloc_convMergingV2_6


nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--merging_type=conv \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_1} \
--fig_save_name=${EXP_DIR_1}.png \
--device=2  > css/${EXP_DIR_1}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--merging_type=conv \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
--fig_save_name=${EXP_DIR_2}.png \
--device=2 > css/${EXP_DIR_2}.log 2>&1 &