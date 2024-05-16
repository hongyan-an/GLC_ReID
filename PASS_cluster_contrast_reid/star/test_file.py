import numpy as np
def read_meta(fn_meta, start_pos=0, verbose=True):
    lb2idxs = {}
    idx2lb = {}
    with open(fn_meta) as f:
        for idx, x in enumerate(f.readlines()[start_pos:]):
            lb = int(x.strip())
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb] += [idx]
            idx2lb[idx] = lb

    inst_num = len(idx2lb)
    cls_num = len(lb2idxs)
    if verbose:
        print('[{}] #cls: {}, #inst: {}'.format(fn_meta, cls_num, inst_num))
    return lb2idxs, idx2lb
def write_meta(fn_meta,labels):
    with open(fn_meta,'w') as f:
        for label in labels:
            f.write(str(label)+'\n')

    print('save done')
def read_probs(path, inst_num, feat_dim, dtype=np.float32, verbose=False):
    assert (inst_num > 0 or inst_num == -1) and feat_dim > 0
    count = -1
    if inst_num > 0:
        count = inst_num * feat_dim
        print('count:',count)
    probs = np.fromfile(path, dtype=dtype, count=count)
    if feat_dim > 1:
        probs = probs.reshape(inst_num, feat_dim)
    if verbose:
        print('[{}] shape: {}'.format(path, probs.shape))
    return probs
labels_s=np.load('target_labels.npy')
write_meta('data/labels/target_labels.meta',labels_s)
# lb2idxs, idx2lb = read_meta('data/labels/target_labels.meta')
# read_probs('data/features/part0_train.bin',len(idx2lb),256,np.float32)
# features=np.load('target_features.npy')
# features.tofile('data/features/target.bin')



