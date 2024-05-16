from utils import knn_faiss,knn_hnsw,l2norm,knns_recall
import numpy as np
def knns_acc(nbrs, idx2lb, lb2idxs):

    recs = []
    cnt = 0
    for idx, (n, _) in enumerate(nbrs):
        lb = idx2lb[idx]
        idxs = lb2idxs[lb]
        n = list(n)
        if len(n) == 1:
            cnt += 1
        s = set(idxs) & set(n)
        recs += [1. * len(s) / len(n)]
    print('there are {} / {} = {:.3f} isolated anchors.'.format(
        cnt, len(nbrs), 1. * cnt / len(nbrs)))
    recall = np.mean(recs)
    return recall
def same_lb(nbrs, idx2lb, lb2idxs):
    recs = []
    cnt = 0
    for idx, (n, _) in enumerate(nbrs[size1:]):
        lb_dict = {}
        for i in n:
            lb=idx2lb[i]
            if lb in lb_dict:
                lb_dict[lb]+=1
            else:
                lb_dict[lb]=1
        print(max(lb_dict.values())/k)
        recs.append(max(lb_dict.values())/k)

    return np.mean(recs)

k = 50
d = 256
nfeat = 10000
np.random.seed(42)

# feats = np.random.random((nfeat, d)).astype('float32')
feats=np.load('source_features.npy').astype('float32')
feats=l2norm(feats)
feats2=np.load('target_features.npy').astype('float32')
feats2=l2norm(feats2)

# feats=np.vstack((feats,feats2))
labels1=np.load('source_labels.npy')
labels2=np.load('target_labels.npy')
idx2lb={}
lb2idxs={}
for i,label in enumerate(labels1):
    idx2lb[i]=label
    if label in lb2idxs:
        lb2idxs[label].append(i)
    else:
        lb2idxs[label]=[i]
size1=labels1.shape[0]
for i,label in enumerate(labels2):
    idx2lb[i+size1]=label
    if label in lb2idxs:
        lb2idxs[label].append(i+size1)
    else:
        lb2idxs[label]=[i+size1]
# labels=np.hstack((labels1,labels2))
# index1 = knn_hnsw(feats, k)
index2 = knn_faiss(feats, k)
print(index2)
# index3 = knn_faiss(feats, k, index_key='Flat')
# index4 = knn_faiss(feats, k, index_key='IVF')
# index5 = knn_faiss(feats, k, index_key='IVF100,PQ32')


# print(same_lb(index3.knns,idx2lb,lb2idxs))
# print(index1.knns[0])
# print('recall:',knns_recall(index3.knns,idx2lb,lb2idxs))
# print('acc:',knns_acc(index3.knns,idx2lb,lb2idxs))
# print('recall:',knns_recall(index4.knns,idx2lb,lb2idxs))
# print('acc:',knns_acc(index4.knns,idx2lb,lb2idxs))
# print('recall:',knns_recall(index5.knns,idx2lb,lb2idxs))
# print('acc:',knns_acc(index5.knns,idx2lb,lb2idxs))
# print(index3.knns[0])
# print(index4.knns[0])
# print(index5.knns[0])