#Single GPU
python train.py --config_file configs/market/vit_small.yml MODEL.PRETRAIN_PATH "/userhome/zhukuan/PASS-reID/pre_trained_model/pass_vit_small_full.pth" OUTPUT_DIR './logs/market/pass_vit_small_full_cat' SOLVER.SEED 42 MODEL.DEVICE_ID '("4")'


