export CUDA_VISIBLE_DEVICES=0
python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py --test --ref_weight=css/experiment/swim_unetr/11.25_default_merging_default_trans_300eps/241_best_metric_model.pth --merging_type='conv' 