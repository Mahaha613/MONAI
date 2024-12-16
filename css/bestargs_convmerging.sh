#!/bin/bash
# eta_min=1e-5
EXP_DIR_1=$(date +%m_%d)_BestArgs_repairIloc_convMerging_7
EXP_DIR_2=$(date +%m_%d)_BestArgs_repairIloc_convMerging_8
EXP_DIR_3=$(date +%m_%d)_BestArgs_repairIloc_convMerging_9
EXP_DIR_4=$(date +%m_%d)_BestArgs_repairIloc_convMerging_10
# EXP_DIR_5=$(date +%m_%d)_BestArgs_repairIloc_convMerging_5


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
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
--fig_save_name=${EXP_DIR_2}.png \
--device=2  > css/${EXP_DIR_2}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--merging_type=conv \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_3} \
--fig_save_name=${EXP_DIR_3}.png \
--device=2  > css/${EXP_DIR_3}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--merging_type=conv \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_4} \
--fig_save_name=${EXP_DIR_4}.png \
--device=2  > css/${EXP_DIR_4}.log 2>&1 &

# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-5 \
# --transforms=css_tr_trs \
# --merging_type=conv \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_5} \
# --fig_save_name=${EXP_DIR_5}.png \
# --device=2  > css/${EXP_DIR_5}.log 2>&1 &

