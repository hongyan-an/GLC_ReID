# VIT-S
CUDA_VISIBLE_DEVICES=2,3 nohup python examples/cluster_contrast_train_usl_glc.py -b 256 -a vit_small -d msmt17 --data-dir '/data/yantianyi/reid_data/' --iters 200 --eps 0.7 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp "/data/yantianyi/pretrain_model/reid/PASS/pretrained_model/pass_vit_small_full.pth" --logs-dir ./log/cluster_contrast_reid_glc/msmt17/pass_vit_small_full_lup_mean_singleneck_glc_12_6 --eval-step 10 --feat-fusion 'mean' &> nohup_usl_msmt_pass_glc_12_6_3_20.out &

