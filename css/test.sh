export CUDA_VISIBLE_DEVICES=2

# Ablation norm&clip:source_data
nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--exp_dir=css/test_result/swim_unetr/$(date +%m_%d)_Ab_norm_clip \
--data_path=BSHD_src_data/image \
--test \
--transforms=my_tr_trs \
--ref_weigh=css/experiment/swim_unetr/11_29_Ab_norm_clip/epoch_433.pth > css/test_$(date +%m_%d)_Ab_norm_clip.log 2>&1 &