import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import argparse
import numpy as np
from mmcv import Config

from star.utils import (create_logger, set_random_seed, rm_suffix,
                   mkdir_if_no_exists)

from star.src.models import build_model
from star.src import train_gcn_reid
from star.test_final_reid import get_pseudo
def run_cluster(source_features,scores,target_features,config='/PASS-reID-main/PASS_cluster_contrast_reid/star/src/configs/cfg_gcn_veri.py',no_cuda=False,work_dir=None,seed=42,load_from=None,resume_from=None):
    # np.save('current_features.npy',features)
    # features=np.load('target_features.npy')
    source_features.numpy().tofile('/PASS-reID-main/PASS_cluster_contrast_reid/star/data/features/source.bin')
    target_features.numpy().tofile('/PASS-reID-main/PASS_cluster_contrast_reid/star/data/features/target.bin')
    # build knn
    if scores is not None:
        np.save('/PASS-reID-main/PASS_cluster_contrast_reid/star/data/source/source.npy',scores)
    cfg = Config.fromfile(config)
    print(cfg)
    # set cuda
    cfg.cuda = not no_cuda and torch.cuda.is_available()

    # set cudnn_benchmark & cudnn_deterministic
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('cudnn_deterministic', False):
        torch.backends.cudnn.deterministic = True

    # update configs according to args
    if not hasattr(cfg, 'work_dir'):
        if work_dir is not None:
            cfg.work_dir = work_dir
        else:
            cfg_name = rm_suffix(os.path.basename(config))
            cfg.work_dir = os.path.join(cfg.prefix+'/work_dir', cfg_name)
    mkdir_if_no_exists(cfg.work_dir, is_folder=True)

    cfg.load_from = load_from
    cfg.resume_from = resume_from

    cfg.gpus = 1
    cfg.distributed = False

    cfg.random_conns = False
    cfg.eval_interim = False
    cfg.save_output = False
    cfg.force = False

    for data in ['train_data', 'test_data']:
        if not hasattr(cfg, data):
            continue
        cfg[data].eval_interim = cfg.eval_interim
        # if not hasattr(cfg[data], 'knn_graph_path') or not os.path.isfile(
        #         cfg[data].knn_graph_path):
        cfg[data].prefix = cfg.prefix
        cfg[data].knn = cfg.knn
        cfg[data].knn_method = cfg.knn_method
        name = 'train_name' if data == 'train_data' else 'test_name'
        cfg[data].name = cfg[name]

    logger = create_logger()

    # set random seeds
    if seed is not None:
        logger.info('Set random seed to {}'.format(seed))
        set_random_seed(seed)

    model = build_model(cfg.model['type'], **cfg.model['kwargs'])
    train_gcn_reid(model,cfg,logger)
    labels,indexes=get_pseudo()



    return labels,indexes

if __name__ == '__main__':
    run_cluster(None)

