# Ablation Spacing,with CropForegroundd
EXP_DIR_1=$(date +%m_%d)_BestArgs_repair_iloc_convMerging_LN
EXP_DIR_2=$(date +%m_%d)_BestArgs_repair_iloc_convMergingV2_LN

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--merging_type=conv \
--transforms=css_tr_trs \
--use_ln \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_1} \
--fig_save_name=${EXP_DIR_1}.png \
--device=1  > css/${EXP_DIR_1}.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--lr=1e-4 \
--eta_min=1e-5 \
--merging_type=conv \
--transforms=css_tr_trs \
--use_ln \
--model=swin_unetr_css_merging \
--exp_dir=css/experiment/swim_unetr/${EXP_DIR_2} \
--fig_save_name=${EXP_DIR_2}.png \
--device=1  > css/${EXP_DIR_2}.log 2>&1 &






