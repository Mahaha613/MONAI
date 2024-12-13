
# Ablation norm&clip:source_data
EXP_DIR=$(date +%m_%d)_Ab_norm_clip_lr0.01
nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--exp_dir=css/test_result/swim_unetr/${EXP_DIR} \
--data_path=BSHD_src_data/image \
--test \
--device="2" \
--transforms=my_tr_trs \
--ref_weigh=css/experiment/swim_unetr/12_02_Ab_norm_clip_lr0.01/epoch_7_dice:0.00020.pth > css/test_${EXP_DIR}.log 2>&1 &