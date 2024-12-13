#!/bin/bash
# eta_min=1e-5
EXP_DIR_1=$(date +%m_%d)_BestArgs_repairIloc_1
EXP_DIR_2=$(date +%m_%d)_BestArgs_repairIloc_2
EXP_DIR_3=$(date +%m_%d)_BestArgs_repairIloc_3
EXP_DIR_4=$(date +%m_%d)_BestArgs_repairIloc_4
EXP_DIR_5=$(date +%m_%d)_BestArgs_repairIloc_5
EXP_DIR_6=$(date +%m_%d)_BestArgs_repairIloc_6
EXP_DIR_7=$(date +%m_%d)_BestArgs_repairIloc_7
EXP_DIR_8=$(date +%m_%d)_BestArgs_repairIloc_8
EXP_DIR_9=$(date +%m_%d)_BestArgs_repairIloc_9
EXP_DIR_10=$(date +%m_%d)_BestArgs_repairIloc_10

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_1} \
--fig_save_name=${EXP_DIR_1}.png \
--device=2  > css/${EXP_DIR_1}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
--fig_save_name=${EXP_DIR_2}.png \
--device=2  > css/${EXP_DIR_2}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_3} \
--fig_save_name=${EXP_DIR_3}.png \
--device=2  > css/${EXP_DIR_3}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_4} \
--fig_save_name=${EXP_DIR_4}.png \
--device=2  > css/${EXP_DIR_4}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_5} \
--fig_save_name=${EXP_DIR_5}.png \
--device=2  > css/${EXP_DIR_5}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-6 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_6} \
--fig_save_name=${EXP_DIR_6}.png \
--device=2  > css/${EXP_DIR_6}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-6 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_7} \
--fig_save_name=${EXP_DIR_7}.png \
--device=2  > css/${EXP_DIR_7}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-6 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_8} \
--fig_save_name=${EXP_DIR_8}.png \
--device=2  > css/${EXP_DIR_8}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-6 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_9} \
--fig_save_name=${EXP_DIR_9}.png \
--device=2  > css/${EXP_DIR_9}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-6 \
--transforms=css_tr_trs \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_10} \
--fig_save_name=${EXP_DIR_10}.png \
--device=2  > css/${EXP_DIR_10}.log 2>&1 &




