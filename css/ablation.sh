export CUDA_VISIBLE_DEVICES=2

# Ablation norm&clip:source_data
# nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
# --data_path=BSHD_src_data/image \
# --lr=1e-3 \
# --exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_norm_clip_lr0.001 \
# --fig_save_path=$(date +%m_%d)_Ab_norm_clip_lr0.001.png \
# --transforms=my_tr_trs > css/$(date +%m_%d)_Ab_norm_clip_lr0.001.log 2>&1 &

# Ablation norm&clip:processed_data
nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--data_path=BSHD_src_data/preprocessed_image \
--lr=1e-3 \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_norm_clip_processed_data_1e-3 \
--fig_save_path=$(date +%m_%d)_Ab_norm_clip_processed_data_1e-3.png \
--transforms=source_tr_trs > css/$(date +%m_%d)_Ab_norm_clip_processed_data_1e-3.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--data_path=BSHD_src_data/preprocessed_image \
--lr=1e-4 \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_norm_clip_processed_data_1e-4 \
--fig_save_path=$(date +%m_%d)_Ab_norm_clip_processed_data_1e-4.png \
--transforms=source_tr_trs > css/$(date +%m_%d)_Ab_norm_clip_processed_data_1e-4.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--data_path=BSHD_src_data/preprocessed_image \
--lr=1e-2 \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_norm_clip_processed_data_1e-2 \
--fig_save_path=$(date +%m_%d)_Ab_norm_clip_processed_data_1e-2.png \
--transforms=source_tr_trs > css/$(date +%m_%d)_Ab_norm_clip_processed_data_1e-2.log 2>&1 &
