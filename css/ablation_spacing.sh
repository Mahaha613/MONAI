# Ablation Spacing,with CropForegroundd
nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--data_path=BSHD_src_data/preprocessed_image \
--lr=1e-3 \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_spacing_lr0.001 \
--fig_save_path=$(date +%m_%d)_Ab_spacing_lr0.001.png \
--spacing="(1.0, 1.0, 2.0)" \
--device=0  > css/$(date +%m_%d)_Ab_spacing_lr0.001.log 2>&1 &


