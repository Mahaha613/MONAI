# Ablation Spacing,without CropForegroundd
nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--data_path=BSHD_src_data/preprocessed_image \
--lr=1e-3 \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_css_trans_lr0.001_spacing1.0 \
--fig_save_path=$(date +%m_%d)_Ab_css_trans_lr0.001_spacing1.0.png \
--spacing="(1.0, 1.0, 2.0)" \
--transforms=css_tr_trs \
--device=1  > css/$(date +%m_%d)_Ab_css_trans_lr0.001_spacing1.0.log 2>&1 &&

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--data_path=BSHD_src_data/preprocessed_image \
--lr=1e-3 \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_Ab_css_trans_lr0.001_spacing1.5 \
--fig_save_path=$(date +%m_%d)_Ab_css_trans_lr0.001_spacing1.5.png \
--spacing="(1.5, 1.5, 2.0)" \
--transforms=css_tr_trs \
--device=1  > css/$(date +%m_%d)_Ab_css_trans_lr0.001_spacing1.5.log 2>&1 &


