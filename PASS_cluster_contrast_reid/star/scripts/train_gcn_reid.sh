cfg_name=cfg_gcn_veri
config=src/configs/$cfg_name.py

export PYTHONPATH=.

# train
CUDA_VISIBLE_DEVICES=4 \
python src/main.py \
    --config $config \
    --phase 'reid'

