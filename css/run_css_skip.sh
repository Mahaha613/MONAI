export CUDA_VISIBLE_DEVICES=1

nohup python /home/xiang/user/user_group/caoshangshang/RushBin/MONAI/css/train.py \
--exp_dir=css/experiment/swim_unetr/$(date +%m_%d)_css_skip_111conv_M4_M1V2_default_trans_500_eps \
--fig_save_path=$(date +%m_%d)_css_skip_111conv_M4_M1V2_default_trans_500_eps.png \
--css_skip \
--use_1x1_conv_for_skip \
--use_css_skip_m4 \
--use_css_skip_m1V2 > css/$(date +%m_%d)_css_skip_111conv_M4_M1V2_default_trans_500_eps.log 2>&1 &
