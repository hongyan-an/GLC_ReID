# VIT-S
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python examples/cluster_contrast_train_usl.py -b 256 -a vit_small -d market1501 --data-dir '/data/yantianyi/reid_data/' --iters 200 --eps 0.6 --self-norm --use-hard --hw-ratio 2 --num-instances 8 -pp "/data/yantianyi/pretrain_model/reid/PASS/pretrained_model/pass_vit_small_full.pth" --logs-dir ./log/cluster_contrast_reid/market/pass_vit_small_cat_singleneck_3_29 --feat-fusion 'cat' &> nohup_usl_market_pass_3_29.out &