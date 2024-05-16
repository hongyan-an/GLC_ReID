# VIT-S
CUDA_VISIBLE_DEVICES=6,7 nohup python examples/cluster_contrast_train_usl_restart.py -b 256 -a vit_small -d msmt17 --data-dir '/data/yantianyi/reid_data/' --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp "/data/yantianyi/pretrain_model/reid/PASS/pretrained_model/pass_vit_small_full.pth" --logs-dir ./log/cluster_contrast_reid_restart/msmt17/pass_vit_small_full_lup_mean_singleneck_restart_15 --eval-step 5 --feat-fusion 'mean' &> nohup_usl_msmt_pass_restart_3_11.out &

