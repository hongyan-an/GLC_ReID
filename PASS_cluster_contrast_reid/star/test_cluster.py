from sklearn.cluster import DBSCAN,KMeans
import numpy as np
import faiss
from evaluation.evaluate import evaluate

from collections import  Counter
def islote(pre_label):
	length=pre_label.shape
	label_length=pre_label.max()
	counts=Counter(pre_label)
	counts_v=np.array(list(counts.values()))
	print('instance num:',length)
	print('label num:', label_length)
	print('instance num == 1:',sum(counts_v==1))
	print('instance num <= 4:', sum(counts_v<5))
	print('instance num avg :', counts_v.mean())
	print('instance num max :', counts_v.max())
	return None

gt_labels=np.load('pretrained_model/gt_labels.npy')
t_features=np.load('target_features.npy')

cluster = faiss.Kmeans(2048, 700, niter=300, verbose=True, gpu=True)
cluster.train(t_features)
_, labels = cluster.index.search(t_features, 1)
k_pre = labels.reshape(-1)
evaluate(gt_labels, k_pre, 'pairwise')
evaluate(gt_labels, k_pre, 'bcubed')
evaluate(gt_labels, k_pre, 'nmi')
islote((k_pre))
# from sklearn.datasets import make_circles
# x,y=make_circles(n_samples=3000,factor=0.3,noise=0.05)
db=DBSCAN(eps=0.6,min_samples=4)
db.fit(t_features)
d_pre=db.labels_
evaluate(gt_labels, d_pre, 'pairwise')
evaluate(gt_labels, d_pre, 'bcubed')
evaluate(gt_labels, d_pre, 'nmi')
islote(d_pre)


