#!/bin/bash

# EXP_DIR_2=$(date +%m_%d)_bestArgs_source_tr_trs_eta_min1e-5
# EXP_DIR_3=$(date +%m_%d)_bestArgs_source_tr_trs_eta_min1e-6
# EXP_DIR_4=$(date +%m_%d)_bestArgs_css_skip
# EXP_DIR_5=$(date +%m_%d)_bestArgs_css_skip_use_1x1_conv_for_skip
# EXP_DIR_6=$(date +%m_%d)_bestArgs_css_skip_use_css_skip_m4
EXP_DIR_7=$(date +%m_%d)_bestArgs_css_skip_use_css_skip_m1V2
EXP_DIR_8=$(date +%m_%d)_bestArgs_css_skip_use_css_skip_m1V2_use_css_skip_m4
EXP_DIR_9=$(date +%m_%d)_bestArgs_css_skip_use_css_skip_m1V2_use_css_skip_m4_use_1x1_conv_for_skip
EXP_DIR_10=$(date +%m_%d)_bestArgs_css_skip_use_css_skip_m1V2_use_1x1_conv_for_skip
EXP_DIR_11=$(date +%m_%d)_bestArgs_css_skip_use_css_skip_m4_use_1x1_conv_for_skip

# css_transB:val_data have a CropForegroundd, but train_data do not have a CropForegroundd

# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-5 \
# --transforms=source_tr_trs \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
# --fig_save_name=${EXP_DIR_2}.png \
# --device=1  > css/${EXP_DIR_2}.log 2>&1 &&

# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-6 \
# --transforms=source_tr_trs \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_3} \
# --fig_save_name=${EXP_DIR_3}.png \
# --device=1  > css/${EXP_DIR_3}.log 2>&1 &&


# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-5 \
# --css_skip \
# --transforms=css_tr_trs \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_4} \
# --fig_save_name=${EXP_DIR_4}.png \
# --device=1  > css/${EXP_DIR_4}.log 2>&1 &&


# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-5 \
# --css_skip \
# --transforms=css_tr_trs \
# --use_1x1_conv_for_skip \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_5} \
# --fig_save_name=${EXP_DIR_5}.png \
# --device=1  > css/${EXP_DIR_5}.log 2>&1 &&

# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-5 \
# --css_skip \
# --transforms=css_tr_trs \
# --use_css_skip_m4 \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_6} \
# --fig_save_name=${EXP_DIR_6}.png \
# --device=1  > css/${EXP_DIR_6}.log 2>&1 &&


nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--css_skip \
--transforms=css_tr_trs \
--use_css_skip_m1V2 \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_7} \
--fig_save_name=${EXP_DIR_7}.png \
--device=1  > css/${EXP_DIR_7}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--css_skip \
--transforms=css_tr_trs \
--use_css_skip_m1V2 \
--use_css_skip_m4 \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_8} \
--fig_save_name=${EXP_DIR_8}.png \
--device=1  > css/${EXP_DIR_8}.log 2>&1 &&


nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--css_skip \
--transforms=css_tr_trs \
--use_css_skip_m1V2 \
--use_css_skip_m4 \
--use_1x1_conv_for_skip \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_9} \
--fig_save_name=${EXP_DIR_9}.png \
--device=1  > css/${EXP_DIR_9}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--css_skip \
--transforms=css_tr_trs \
--use_css_skip_m1V2 \
--use_1x1_conv_for_skip \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_10} \
--fig_save_name=${EXP_DIR_10}.png \
--device=1  > css/${EXP_DIR_10}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--css_skip \
--transforms=css_tr_trs \
--use_css_skip_m4 \
--use_1x1_conv_for_skip \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_11} \
--fig_save_name=${EXP_DIR_11}.png \
--device=1  > css/${EXP_DIR_11}.log 2>&1 &