import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
import time
from sklearn.metrics import f1_score
import os
from scipy import io
from param import root, results_dir, ds, data_names, TRAIN_BATCH_SIZE
from  modules import ORDisModel, SSHIDA
from torchvision.models import resnet18
from backbone_SITS_CNN import CNN3D_Encoder, CNN2D_Encoder
from sklearn.metrics import confusion_matrix

def evaluation_OUR(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        if y_batch.shape[0] == TRAIN_BATCH_SIZE:
            print(f"x = {x_batch.shape}, y = {y_batch.shape}")
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = None
            _, _ ,_, pred = model.forward_test_target(x_batch)
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

def evaluation_SSHIDA(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        _, pred = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

def evaluation_FIXMATCH(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels

def evaluation_TG(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(x_batch)
        pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


##########################
# MAIN FUNCTION: TRAINING
##########################
#def main():
def train_and_eval(ds_path, nsamples, nsplit, ds_idx, source_idx, gpu, method, model_path):
    '''

    :param nsamples:
    :param nsplit:
    :param ds_idx: Index of the dataset [0, 1, 2, 3]
    :param source_idx: Index of the source modality
    :return:
    '''

    evaluation = {
        "FIXMATCH": evaluation_FIXMATCH,
        "SSHIDA": evaluation_SSHIDA,
        "TG-ONLY": evaluation_TG,
        "OUR": evaluation_OUR,
        "OUR-GRL-EMA": evaluation_OUR,
    }

    #gpu = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    year_target = "2021"

    source_prefix = data_names[ds[ds_idx]][source_idx]
    target_prefix = data_names[ds[ds_idx]][source_idx - 1]

    target_data = np.load(os.path.join(ds_path, ds[ds_idx], year_target, f"{target_prefix}_data_filtered.npy"))
    target_label = np.load(os.path.join(ds_path, ds[ds_idx], year_target, f"{target_prefix}_label_filtered.npy"))
    #target_label = np.where(target_label <= 2, 0, target_label - 2)


    sys.stdout.flush()
    train_target_idx = np.load( os.path.join(ds_path, ds[ds_idx], year_target, "train_idx", f"{target_prefix}_{nsplit}_{nsamples}_train_idx.npy") )
    test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)

    #test_target_data = target_data[train_target_idx]
    #test_target_label = target_label[train_target_idx]

    test_target_data = target_data[test_target_idx]
    test_target_label = target_label[test_target_idx]

    counts = np.bincount(test_target_label, minlength=6)
    print(f"Occorrenze: {counts}")

    #print(f"Len train: {train_target_label.shape}, Len test: {test_target_label.shape}")

    test_target_data_unl = target_data[test_target_idx]

    n_classes = len(np.unique(target_label))

    if method == "FIXMATCH" or method == "TG-ONLY":
        print(f"Model = ResNet")
        #model = CNN3D_Encoder(n_filters=target_data.shape[1], n_classes=n_classes, drop=0.0)
        model = CNN2D_Encoder(n_filters=target_data.shape[1] * target_data.shape[2], n_classes=n_classes, drop=0.0)
        model = model.to(device)

    print("Loading model...")
    print(f"Path: {model_path}")
    state_dict = torch.load(model_path, weights_only=True)  # Replace 'model_weights.pth' with your file path
    #state_dict = torch.load(model_path)  # Replace 'model_weights.pth' with your file path
    model.load_state_dict(state_dict, strict=False)
    print("Loaded!")

    model = model.to(device)

    #DATALOADER TARGET TEST
    x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
    y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
    dataset_test_target = TensorDataset(x_test_target, y_test_target)
    dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

    pred_valid, labels_valid = evaluation_TG(model, dataloader_test_target, device)

    # Calcola la matrice di confusione
    cm_normalized = confusion_matrix(labels_valid, pred_valid)
    #cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalizza per riga
    #print(cm_normalized)

    f1_val = f1_score(labels_valid, pred_valid, average=None)
    f1_val_mean = f1_score(labels_valid, pred_valid, average="weighted")

    return 100*f1_val, 100*f1_val_mean, cm_normalized

if __name__ == "__main__":
    #main()

    ds_idx = 2
    source_idx = 0
    #methods = ["FIXMATCH", "TG-ONLY", "SSHIDA", "OUR"]
    method = "TG-ONLY"
    gpu = "1"
    nsplit = 2
    nsamples = 50

    root_ = "/home/giuseppe"
    #root_ = "/home/giuseppe.guarino"
    ds_path = "/home/giuseppe/SITS"

    model_paths = {
        "TG-ONLY": root_ + "/A-SHeDD/Results_6_2D_light/Koumbia_6/ResNet-18/TG-ONLY/models",
    }

    table = []
    out_dir = os.path.join("Results_6_2D_light", ds[ds_idx], "ResNet-18", "TG-ONLY")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Method: {method}")
    sys.stdout.flush()

    model_path = os.path.join(model_paths[method], data_names[ds[ds_idx]][source_idx],
                              f"{data_names[ds[ds_idx]][source_idx]}_{nsplit}_{nsamples}.pth")

    print(f"Processing {ds[ds_idx]} Source = {data_names[ds[ds_idx]][source_idx]}. nsplit = {nsplit}, nsamples = {nsamples}")
    sys.stdout.flush()
    f1_val, f1_avg, cm = train_and_eval(ds_path, nsamples, nsplit, ds_idx, source_idx, gpu, method, model_path)
    table.append([ds_idx, source_idx, nsplit, nsamples, f1_val, f1_avg])
    io.savemat(os.path.join(out_dir, f'F1_{method}_{ds[ds_idx]}_{data_names[ds[ds_idx]][source_idx]}_{nsplit}.mat'),
               {'RESULTS': table, 'CM': cm})