import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
from backbone_resnet import ORDisModel, ORDisModelGRL
import time
from sklearn.metrics import f1_score
import torch.nn.functional as F
from functions import MyDataset_Unl, MyDataset, cumulate_EMA, transform_1, transform_2
import os
from scipy import io
from param import root, results_dir, ds, data_names, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA, MU
from tqdm import tqdm
import competitors.fixmatch as fixmatch
import competitors.SSHIDA as sshida
import competitors.target_only as tg_only
import PSHeDD
import main
from  modules import ORDisModel, SSHIDA
from torchvision.models import resnet18

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
def train_and_eval(nsamples, nsplit, ds_idx, source_idx, gpu, method, model_path):
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

    source_prefix = data_names[ds[ds_idx]][source_idx]
    target_prefix = data_names[ds[ds_idx]][source_idx - 1]

    source_data = np.load(os.path.join(root, ds[ds_idx], f"{source_prefix}_data_filtered.npy"))
    target_data = np.load(os.path.join(root, ds[ds_idx], f"{target_prefix}_data_filtered.npy"))
    source_label = np.load(os.path.join(root, ds[ds_idx], f"{source_prefix}_label_filtered.npy"))
    target_label = np.load(os.path.join(root, ds[ds_idx], f"{target_prefix}_label_filtered.npy"))

    sys.stdout.flush()
    train_target_idx = np.load( os.path.join(root, ds[ds_idx], "train_idx", f"{target_prefix}_{nsplit}_{nsamples}_train_idx.npy") )
    test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)

    train_target_data = target_data[train_target_idx]
    train_target_label = target_label[train_target_idx]

    test_target_data = target_data[test_target_idx]
    test_target_label = target_label[test_target_idx]

    test_target_data_unl = target_data[test_target_idx]

    n_classes = len(np.unique(source_label))

    if method == "FIXMATCH" or method == "TG-ONLY":
        print(f"Model = ResNet")
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(train_target_data.shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        model._modules["fc"] = nn.Linear(in_features=512, out_features=n_classes)
    elif method == "OUR" or method == "OUR-GRL-EMA":
        print(f"Model = ORDisModel")
        model = ORDisModel(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)
    else:
        print(f"Model = SSHIDA")
        model = SSHIDA(input_channel_source=source_data.shape[1], input_channel_target=target_data.shape[1], num_classes=n_classes)


    #checkpoint = torch.load(model_path, weights_only=True)
    #model.load_state_dict(torch.load(model_path, weights_only=True))
    #model.load_state_dict(model_path)

    print("Loading model...")
    state_dict = torch.load(model_path, weights_only=True)  # Replace 'model_weights.pth' with your file path
    model.load_state_dict(state_dict, strict=False)
    print("Loaded!")

    model = model.to(device)

    #DATALOADER TARGET TEST
    x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
    y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
    dataset_test_target = TensorDataset(x_test_target, y_test_target)
    dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

    pred_valid, labels_valid = evaluation[method](model, dataloader_test_target, device)
    f1_val = f1_score(labels_valid, pred_valid, average=None)
    f1_val_mean = f1_score(labels_valid, pred_valid, average="weighted")

    return 100*f1_val, 100*f1_val_mean

if __name__ == "__main__":
    #main()

    ds_idx = 4
    source_idx = 1
    #methods = ["FIXMATCH", "TG-ONLY", "SSHIDA", "OUR"]
    method = "OUR-GRL-EMA"
    gpu = "0"
    nsplit = 5
    nsamples = 10

    root_ = "/home/giuseppe"
    #root_ = "/home/giuseppe.guarino"

    model_paths = {
        "FIXMATCH": root_ + "/SHeDD/results/EUROSAT-MS-SAR/FIXMATCH/models",
        "SSHIDA": root_ + "/SHeDD/results/EUROSAT-MS-SAR/SSHIDA/models",
        "TG-ONLY": root_ + "/SHeDD/results/EUROSAT-MS-SAR/TG-ONLY/models",
        "OUR-GRL-EMA": root_ + "/SHeDD/results/EUROSAT-MS-SAR/OUR-GRL-EMA/models",
        "OUR": root_ + "/SHeDD/results/EUROSAT-MS-SAR/OUR/models",
    }

    table = []
    out_dir = os.path.join(results_dir, ds[ds_idx], method)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Method: {method}")
    sys.stdout.flush()

    model_path = os.path.join(model_paths[method], data_names[ds[ds_idx]][source_idx],
                              f"{data_names[ds[ds_idx]][source_idx]}_{nsplit}_{nsamples}.pth")

    print(f"Processing {ds[ds_idx]} Source = {data_names[ds[ds_idx]][source_idx]}. nsplit = {nsplit}, nsamples = {nsamples}")
    sys.stdout.flush()
    f1_val, f1_avg = train_and_eval(nsamples, nsplit, ds_idx, source_idx, gpu, method, model_path)
    table.append([ds_idx, source_idx, nsplit, nsamples, f1_val, f1_avg])
    io.savemat(os.path.join(out_dir, f'F1_{method}_{ds[ds_idx]}_{data_names[ds[ds_idx]][source_idx]}.mat'),
               {'RESULTS': table})