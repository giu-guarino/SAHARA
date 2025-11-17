import numpy as np
from sklearn.utils import shuffle
import os
from param import root, results_dir, ds, data_names, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA

def rescale(data):
    min_ = np.percentile(data, 2)
    max_ = np.percentile(data, 98)
    return np.clip( (data - min_) / (max_ - min_), 0, 1.)

def getIdxVal(sub_hashCl2idx, val):
    idx = []
    for k in sub_hashCl2idx.keys():
        temp = sub_hashCl2idx[k]
        idx.append(temp[0:val])
    return np.concatenate(idx, axis=0)


def get_idxPerClass(hashCl2idx, max_val):
    '''
    :param hashCl2idx:
    :param max_val: The number of labeled data to consider
    :return:
    '''
    sub_hashCl2idx = {}
    for k in hashCl2idx.keys():
        temp = hashCl2idx[k]
        temp = shuffle(temp)
        sub_hashCl2idx[k] = temp[0:max_val]
    return sub_hashCl2idx


def extractWriteTrainIdx(root, nrepeat, nsample_list, hashCl2idx, dataset, modality):
    max_val = nsample_list[-1]
    for i in range(nrepeat):
        sub_hashCl2idx = get_idxPerClass(hashCl2idx, max_val)
        for val in nsample_list:
            idx = getIdxVal(sub_hashCl2idx, val)
            np.save(os.path.join(root, dataset, "train_idx", "%s_%d_%d_train_idx.npy"%(modality,i,val)), idx)


def getHash2classes(labels):
    hashCl2idx = {}
    for v in np.unique(labels):
        idx = np.where(labels == v)[0] #== np.asarray(labels == v).nonzero()[0] (ritorna gli indici che rispettano la condizione)
        idx = shuffle(idx)
        hashCl2idx[v] = idx
    return hashCl2idx
        
def writeFilteredData(root, dataset, modality, data, label):
    np.save(os.path.join(root, dataset, "%s_data_filtered.npy"%modality), data)
    np.save(os.path.join(root, dataset, "%s_label_filtered.npy"%modality), label)

#root = "/home/giuseppe.guarino/Datasets/Datasets_INRIA/"
#ds = ["AV-MNIST", "CREMA-D", "SUNRGBD", "TRISTAR"]
#ds_idx = 0
#data_names = {
#    "AV-MNIST": ["MNIST", "SPECTRO"],
#    "CREMA-D": ["VIDEO", "SPECTRO"],
#    "SUNRGBD": ["DEPTH", "RGB"],
#    "TRISTAR": ["DEPTH", "THERMAL"],
#}

for ds_idx in [7]:

    print(f"Processing {ds[ds_idx]}")

    out_dir = os.path.join(root, ds[ds_idx], "train_idx")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #READ DATA
    #data_sar = np.load(os.path.join(root, "SAR/SAR_data.npy"))
    #data_opt = np.load(os.path.join(root, "MS/MS_data.npy")).astype("float32")

    data_mod1 = np.load(os.path.join(root, ds[ds_idx], f"{data_names[ds[ds_idx]][0]}_data_normalized.npy")).astype("float32")
    data_mod2 = np.load(os.path.join(root, ds[ds_idx], f"{data_names[ds[ds_idx]][1]}_data_normalized.npy")).astype("float32")

    # labels = np.load(os.path.join(root, ds[ds_idx], f"labels.npy"))

    #SELECT UNCORRELATED DATA
    if ds[ds_idx] == "AV-MNIST" or ds[ds_idx] == "CREMA-D":
        labels = np.load(os.path.join(root, ds[ds_idx], f"labels.npy"))

        label_mod1 = labels
        label_mod2 = labels

    elif ds[ds_idx] == "RESISC45_EURO":
        label_mod1 = np.load(os.path.join(root, ds[ds_idx], f"{data_names[ds[ds_idx]][0]}_labels.npy"))
        label_mod2 = np.load(os.path.join(root, ds[ds_idx], f"{data_names[ds[ds_idx]][1]}_labels.npy"))

    else:
        labels = np.load(os.path.join(root, ds[ds_idx], f"labels.npy"))

        mod1_idx = np.arange(0, labels.shape[0], 2)
        mod2_idx = np.arange(1, labels.shape[0], 2)

        data_mod1 = data_mod1[mod1_idx]
        label_mod1 = labels[mod1_idx]

        data_mod2 = data_mod2[mod2_idx]
        label_mod2 = labels[mod2_idx]


    #RESCALE DATA BETWEEN 0 and 1 per band
    #for i in range(data_sar.shape[-1]):
    #    data_sar[:,:,:,i] = rescale(data_sar[:,:,:,i])

    #for i in range(data_opt.shape[-1]):
    #    data_opt[:,:,:,i] = rescale(data_opt[:,:,:,i])

    #SWAP DIMENSION : [N,H,W,C] -> [N,C,H,W]
    #data_sar = np.moveaxis(data_sar, (0,1,2,3), (0,2,3,1))
    #data_opt = np.moveaxis(data_opt, (0,1,2,3), (0,2,3,1))

    #WRITE DATA
    writeFilteredData(root, ds[ds_idx], data_names[ds[ds_idx]][0], data_mod1, label_mod1)
    writeFilteredData(root, ds[ds_idx], data_names[ds[ds_idx]][1], data_mod2, label_mod2)

    mod1_hashCl2idx = getHash2classes(label_mod1)
    mod2_hashCl2idx = getHash2classes(label_mod2)

    #EXTRACT 10 time TRAIN IDX INCREASING THE NUMBER OF SAMLPE PER CLASS FROM 50 TO 400
    nrepeat = 10
    nsample_list = [5, 10, 25, 50]
    extractWriteTrainIdx(root, nrepeat, nsample_list, mod1_hashCl2idx, ds[ds_idx], data_names[ds[ds_idx]][0])
    extractWriteTrainIdx(root, nrepeat, nsample_list, mod2_hashCl2idx, ds[ds_idx], data_names[ds[ds_idx]][1])



