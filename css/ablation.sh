export CUDA_VISIBLE_DEVICES=1

# Ablation norm&clip:source_data
nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--data_path=BSHD_src_data/image \
--lr=1e-2 \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_norm_clip_lr0.01 \
--fig_save_path=$(date +%m_%d)_Ab_norm_clip_lr0.01.png \
--transforms=my_tr_trs > css/$(date +%m_%d)_Ab_norm_clip_lr0.01.log 2>&1 &

# Ablation norm&clip:processed_data
# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --data_path=BSHD_src_data/preprocessed_image \
# --exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_norm_clip \
# --fig_save_path=$(date +%m_%d)_Ab_norm_clip.png \
# --transforms=my_tr_trs > css/(date +%m_%d)_Ab_norm_clip.log 2>&1 &
