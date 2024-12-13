# Ablation Spacing,with CropForegroundd
EXP_DIR_1=$(date +%m_%d)_BestArgs_convMerging
EXP_DIR_2=$(date +%m_%d)_BestArgs_maxpoolMerging
EXP_DIR_3=$(date +%m_%d)_BestArgs_avgpoolMerging
EXP_DIR_4=$(date +%m_%d)_BestArgs_maxavgpoolMerging

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--merging_type=conv \
--transforms=css_tr_trs \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_1} \
--fig_save_name=${EXP_DIR_1}.png \
--device=1  > css/${EXP_DIR_1}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--merging_type=maxpool \
--transforms=css_tr_trs \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
--fig_save_name=${EXP_DIR_2}.png \
--device=1  > css/${EXP_DIR_2}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--merging_type=avgpool \
--transforms=css_tr_trs \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_3} \
--fig_save_name=${EXP_DIR_3}.png \
--device=1  > css/${EXP_DIR_3}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--merging_type=maxavgpool \
--transforms=css_tr_trs \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_4} \
--fig_save_name=${EXP_DIR_4}.png \
--device=1  > css/${EXP_DIR_4}.log 2>&1 &

# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-7 \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
# --fig_save_name=${EXP_DIR_2}.png \
# --device=2  > css/${EXP_DIR_2}.log 2>&1 &&


# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-5 \
# --eta_min=1e-6 \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_3} \
# --fig_save_name=${EXP_DIR_3}.png \
# --device=2  > css/${EXP_DIR_3}.log 2>&1 &&

# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --lr=1e-4 \
# --eta_min=1e-7 \
# --exp_dir=css/experiment/swim_unetr/${EXP_DIR_4} \
# --fig_save_name=${EXP_DIR_4}.png \
# --device=2  > css/${EXP_DIR_4}.log 2>&1 &




