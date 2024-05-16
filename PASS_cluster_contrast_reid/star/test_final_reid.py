import torch
import linecache
import numpy as np


from star.evaluation.evaluate import evaluate
import os
from star.src.models.gcn import HEAD, HEAD_test
from star.src.models import build_model
from mmcv import Config
from star.src.datasets import build_dataset
from star.utils import sparse_mx_to_torch_sparse_tensor, build_knns, fast_knns2spmat, build_symmetric_adj, row_normalize,mkdir_if_no_exists,Timer
import torch.nn as nn
from star.utils.misc import l2norm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import  Counter
def islote(pre_label):
	lb2idxs = {}
	lb2index={}
	count=0
	for i, label in enumerate(pre_label):
		if label in lb2idxs:
			lb2idxs[label].append(i)
		else:
			lb2idxs[label] = [i]


	length=pre_label.shape
	label_length=pre_label.max()
	labels_ = np.arange(0, label_length + 1)
	counts=Counter(pre_label)
	counts_v=np.array(list(counts.values()))
	counts_v_5=counts_v[counts_v<5]
	print('instance num:',length)
	print('label num:', label_length)
	print('instance num == 1:',sum(counts_v==1))
	print('instance_id num <= 4:', sum(counts_v<5))
	print('instance num <= 4:', sum(counts_v_5))
	print('instance num avg :', counts_v.mean())
	print('instance num max :', counts_v.max())
	reserve_id = []
	for k, v in lb2idxs.items():
		if len(v) > 4:
			reserve_id = reserve_id + v
	return np.array(reserve_id)


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while (u != parent[u]):
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id
def get_pseudo():
	flag=1
	k_=50
	if flag == 1:

		cfg = Config.fromfile("PASS-reID-main/PASS_cluster_contrast_reid/star/src/configs/cfg_gcn_veri.py")
		cfg.eval_interim = False
		cfg['test_data'].eval_interim = cfg.eval_interim
		# if not hasattr(cfg[data], 'knn_graph_path') or not os.path.isfile(
		#         cfg[data].knn_graph_path):
		cfg['test_data'].prefix = cfg.prefix
		cfg['test_data'].knn = cfg.knn
		cfg['test_data'].knn_method = cfg.knn_method

		cfg['test_data'].name = cfg['test_name']
		target = "target"
		# feature_path = cfg.star_path+'/data/features"
		feature_path=cfg.star_path+'data/features'
		#model_path_list=['train_model_sample7']
		#backbone_index=['4299']c

		for model_i in [0]:
			model_i = int(model_i)
			model_path = cfg.star_path+"/src/train_model"
			print('model_path',model_path)
			# backbone_name = "Backbone_1_599_50.pth"
			# HEAD_name = "Head_1_599_50.pth"
                        # change Backbone_Epoch_1_batch_199.pth form epoch1 to epoch2
			backbone_name = "Backbone_Epoch_2_batch_199.pth"
			HEAD_name = "Head_Epoch_2_batch_199.pth"
			use_cuda = True
			knn_path = cfg.star_path+"/data/knns/" + target + "/faiss_k_50.npz"
			use_gcn = True

			if use_gcn:
				knns = np.load(knn_path, allow_pickle=True)['data']
				nbrs = knns[:, 0, :]
				dists = knns[:, 1, :]
				edges = []
				score = []
				inst_num = knns.shape[0]
				print("inst_num:", inst_num)

				feature_path = os.path.join(feature_path, target)

				# print(**cfg.model['kwargs'])
				model = build_model('gcn', **cfg.model['kwargs'])
				model.load_state_dict(torch.load(os.path.join(model_path, backbone_name)))
				HEAD_test1 = HEAD_test(nhid=512)
				HEAD_test1.load_state_dict(torch.load(os.path.join(model_path, HEAD_name)), False)

				with Timer('build dataset'):
					for k, v in cfg.model['kwargs'].items():
						setattr(cfg.test_data, k, v)
					dataset = build_dataset(cfg.model['type'], cfg.test_data)

				features = torch.FloatTensor(dataset.features)
				adj = sparse_mx_to_torch_sparse_tensor(dataset.adj)
				if not dataset.ignore_label:
					labels = torch.FloatTensor(dataset.gt_labels)
				samples_num=features.shape[0]
				pair_a = []
				pair_b = []
				pair_a_new = []
				pair_b_new = []
				for i in range(inst_num):
					pair_a.extend([int(i)] * k_)
					pair_b.extend([int(j) for j in nbrs[i]])
				for i in range(len(pair_a)):
					if pair_a[i] != pair_b[i]:
						pair_a_new.extend([pair_a[i]])
						pair_b_new.extend([pair_b[i]])
				pair_a = pair_a_new
				pair_b = pair_b_new
				print(len(pair_a))
				inst_num = len(pair_a)
				if use_cuda:
					model.cuda()
					HEAD_test1.cuda()
					features = features.cuda()
					adj = adj.cuda()
					labels = labels.cuda()

				model.eval()
				HEAD_test1.eval()
				test_data = [[features, adj, labels]]

				for threshold1 in [0.6]:
					with Timer('Inference'):
						with Timer('First-0 step'):
							with torch.no_grad():
								output_feature = model(test_data[0])

								patch_num = 65
								patch_size = int(inst_num / patch_num)
								for i in range(patch_num):
									id1 = pair_a[i * patch_size:(i + 1) * patch_size]
									id2 = pair_b[i * patch_size:(i + 1) * patch_size]
									score_ = HEAD_test1(output_feature[id1],output_feature[id2])
									score_ = np.array(score_)
									idx = np.where(score_ > threshold1)[0].tolist()
									#score.extend(score_[idx].tolist())
									id1 = np.array(id1)
									id2 = np.array(id2)
									id1 = np.array([id1[idx].tolist()])
									id2 = np.array([id2[idx].tolist()])
									edges.extend(np.concatenate([id1, id2], 0).transpose().tolist())
									#print('patch id:',i)

							value=[1]*len(edges)
							edges=np.array(edges)

						with Timer('First step'):
							adj2 = csr_matrix((value, (edges[:,0].tolist(), edges[:,1].tolist())), shape=(samples_num, samples_num))
							link_num = np.array(adj2.sum(axis=1))
							common_link = adj2.dot(adj2)

						for threshold2 in [0.6]:
							with Timer('Second step'):
								edges_new = []
								edges = np.array(edges)
								share_num = common_link[edges[:,0].tolist(), edges[:,1].tolist()].tolist()[0]
								edges = edges.tolist()

								for i in range(len(edges)):
									if ((link_num[edges[i][0]]) != 0) & ((link_num[edges[i][1]]) != 0):
										if max((share_num[i])/link_num[edges[i][0]],(share_num[i])/link_num[edges[i][1]])>threshold2:
											edges_new.append(edges[i])
									if i%10000000==0:
										print(i)

							with Timer('Last step'):
								pre_labels = edge_to_connected_graph(edges_new, samples_num)
							gt_labels = np.load(cfg.star_path+'/pretrained_model/gt_labels.npy')
							print('the threshold1 is:{}'.format(threshold1))
							print('the threshold2 is:{}'.format(threshold2))
							evaluate(gt_labels, pre_labels, 'pairwise')
							evaluate(gt_labels, pre_labels, 'bcubed')
							evaluate(gt_labels, pre_labels, 'nmi')
							islote(gt_labels)
							# print(length,counts.shape)
							indexes=islote(pre_labels)
							# print(length, counts.shape)


							evaluate(gt_labels[indexes], pre_labels[indexes], 'pairwise')
							evaluate(gt_labels[indexes], pre_labels[indexes], 'bcubed')
							evaluate(gt_labels[indexes], pre_labels[indexes], 'nmi')

	return pre_labels,indexes

if __name__ == '__main__':
	get_pseudo()

