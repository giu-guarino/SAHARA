import torch
import torch.nn as nn
import sys
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.utils import shuffle
import backbone_resnet
import backbone_vit
from modules import FC_Classifier_2L, GradientReversal
import time
from sklearn.metrics import f1_score
import torch.nn.functional as F
from torch.autograd import grad
from functions import MyDataset_Unl, MyDataset, cumulate_EMA, transform
import os
from scipy import io
from param import ds, data_names, TRAIN_BATCH_SIZE, LEARNING_RATE, MOMENTUM_EMA, EPOCHS, TH_FIXMATCH, WARM_UP_EPOCH_EMA, TV_param
from tqdm import tqdm
import gc
from torch.amp import autocast, GradScaler

ITER_DC = 10 # Inner iterations for domain-critic optimization
ITER_CLF = 1 # Inner iterations for classifier optimization
ALPHA = 1.
ALPHA_FM = 0.9


def evaluation(model, dataloader, device):
    model.eval()
    tot_pred = []
    tot_labels = []
    for x_batch, y_batch in dataloader:
        if y_batch.shape[0] == TRAIN_BATCH_SIZE:
            #print(f"x = {x_batch.shape}, y = {y_batch.shape}")
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = None
            _, _ ,_, pred = model.forward_test_target(x_batch)
            pred_npy = np.argmax(pred.cpu().detach().numpy(), axis=1)
            tot_pred.append( pred_npy )
            tot_labels.append( y_batch.cpu().detach().numpy())
            torch.cuda.empty_cache()
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)
    return tot_pred, tot_labels


def gradient_penalty(critic, h_s, h_t, device):
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad



##########################
# MAIN FUNCTION: TRAINING
##########################
def train_and_eval(ds_path, out_dir, nsamples, nsplit, ds_idx, source_idx, gpu, backbone):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    source_prefix = data_names[ds[ds_idx]][source_idx]
    target_prefix = data_names[ds[ds_idx]][source_idx - 1]

    source_data = np.load(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_data_filtered.npy"))
    target_data = np.load(os.path.join(ds_path, ds[ds_idx], f"{target_prefix}_data_filtered.npy"))
    source_label = np.load(os.path.join(ds_path, ds[ds_idx], f"{source_prefix}_label_filtered.npy"))
    target_label = np.load(os.path.join(ds_path, ds[ds_idx], f"{target_prefix}_label_filtered.npy"))

    train_target_idx = np.load( os.path.join(ds_path, ds[ds_idx], "train_idx", f"{target_prefix}_{nsplit}_{nsamples}_train_idx.npy") )
    test_target_idx = np.setdiff1d(np.arange(target_data.shape[0]), train_target_idx)

    train_target_data = target_data[train_target_idx]
    train_target_label = target_label[train_target_idx]

    test_target_data = target_data[test_target_idx]
    test_target_label = target_label[test_target_idx]

    test_target_data_unl = target_data[test_target_idx]

    n_classes = len(np.unique(source_label))
    THRS = torch.ones(n_classes, device=device) * (1.0 / n_classes)

    TR_BATCH_SIZE = np.minimum(int(n_classes * nsamples), TRAIN_BATCH_SIZE)
    TR_BATCH_SIZE = int(TR_BATCH_SIZE)

    source_data, source_label = shuffle(source_data, source_label)
    train_target_data, train_target_label = shuffle(train_target_data, train_target_label)

    #DATALOADER SOURCE
    x_train_source = torch.tensor(source_data, dtype=torch.float32)
    y_train_source = torch.tensor(source_label, dtype=torch.int64)

    #dataset_source = TensorDataset(x_train_source, y_train_source)
    dataset_source = MyDataset(x_train_source, y_train_source, transform=transform)
    dataloader_source = DataLoader(dataset_source, shuffle=True, batch_size=TR_BATCH_SIZE)

    input_channel_source, img_size_source = source_data.shape[1], source_data.shape[2]
    input_channel_target, img_size_target = target_data.shape[1], target_data.shape[2]

    del source_data, source_label
    gc.collect()

    #DATALOADER TARGET TRAIN
    x_train_target = torch.tensor(train_target_data, dtype=torch.float32)
    y_train_target = torch.tensor(train_target_label, dtype=torch.int64)

    del train_target_data, train_target_label
    gc.collect()

    dataset_train_target = MyDataset(x_train_target, y_train_target, transform=transform)
    dataloader_train_target = DataLoader(dataset_train_target, shuffle=True, batch_size=TR_BATCH_SIZE//2)

    #DATALOADER TARGET UNLABELLED
    x_train_target_unl = torch.tensor(test_target_data_unl, dtype=torch.float32)

    dataset_train_target_unl = MyDataset_Unl(x_train_target_unl, transform=transform)
    dataloader_train_target_unl = DataLoader(dataset_train_target_unl, shuffle=True, batch_size=TR_BATCH_SIZE//2)

    del x_train_target_unl  # test_target_data sarà eliminato più avanti
    gc.collect()

    #DATALOADER TARGET TEST
    x_test_target = torch.tensor(test_target_data, dtype=torch.float32)
    y_test_target = torch.tensor(test_target_label, dtype=torch.int64)
    dataset_test_target = TensorDataset(x_test_target, y_test_target)
    dataloader_test_target = DataLoader(dataset_test_target, shuffle=False, batch_size=TRAIN_BATCH_SIZE)

    # Liberare memoria
    del test_target_data, test_target_label, target_data
    gc.collect()

    if backbone == "CNN":
        model = backbone_resnet.SAHARA(input_channel_source=input_channel_source, input_channel_target=input_channel_target, num_classes=n_classes)
        model = model.to(device)

        domain_cl = FC_Classifier_2L(256*n_classes, 2, 128, act=nn.ReLU())
        domain_cl = domain_cl.to(device)

    else:
        model = backbone_vit.SAHARA(img_size_source=img_size_source, in_chans_source=input_channel_source,
                         img_size_target=img_size_target, in_chans_target=input_channel_target,
                         num_classes=n_classes,
                         embed_dims=TV_param["embed_dims"], depths=TV_param["depths"],
                         num_heads=TV_param["num_heads"], window_sizes=TV_param["window_sizes"],
                         drop_path_rate=TV_param["drop_path_rate"]
                         )
        model = model.to(device)

        domain_cl = FC_Classifier_2L((TV_param["embed_dims"][-1]//2)*n_classes, 2, 128, act=nn.ReLU())
        domain_cl = domain_cl.to(device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)

    grl = GradientReversal(ALPHA).to(device)

    pbar = tqdm(range(EPOCHS))
    scaler = GradScaler(device)

    ema_weights = None

    for epoch in pbar:
        pbar.set_description('Epoch %d/%d' % (epoch + 1, EPOCHS))
        start = time.time()
        model.train()
        tot_loss = 0.0
        tot_ortho_loss = 0.0
        tot_fixmatch_loss = 0.0
        den = 0

        for x_batch_source, y_batch_source in dataloader_source:
            if x_batch_source.shape[0] < TR_BATCH_SIZE:
                continue  # To avoid errors on pairing source/target samples

            optimizer.zero_grad()
            x_batch_target, y_batch_target = next(iter(dataloader_train_target))
            x_batch_target_unl, x_batch_target_unl_aug = next(iter(dataloader_train_target_unl))

            x_batch_source = x_batch_source.to(device)
            y_batch_source = y_batch_source.to(device)

            x_batch_target = x_batch_target.to(device)
            y_batch_target = y_batch_target.to(device)

            x_batch_target_unl = x_batch_target_unl.to(device)
            x_batch_target_unl_aug = x_batch_target_unl_aug.to(device)

            with autocast(device):

                emb_source_inv, emb_source_spec, dom_source_cl, task_source_cl, emb_target_inv, emb_target_spec, dom_target_cl, task_target_cl = model(
                    [x_batch_source, x_batch_target])

                pred_task = torch.cat([task_source_cl, task_target_cl], dim=0)
                pred_dom = torch.cat([dom_source_cl, dom_target_cl], dim=0)
                y_batch = torch.cat([y_batch_source, y_batch_target], dim=0)
                y_batch_dom = torch.cat([torch.zeros_like(y_batch_source), torch.ones_like(y_batch_target)], dim=0)

                loss_pred = loss_fn(pred_task, y_batch)
                loss_dom = loss_fn( pred_dom, y_batch_dom)

                model.target.train()
                unl_target_inv, unl_target_spec, pred_unl_target_dom, pred_unl_target = model.forward_source(x_batch_target_unl, 1)
                unl_target_aug_inv, unl_target_aug_spec, pred_unl_target_strong_dom, pred_unl_target_strong = model.forward_source(x_batch_target_unl_aug, 1)

                # Training loop
                with torch.no_grad():
                    pseudo_labels = torch.softmax(pred_unl_target, dim=1)
                    max_probs, targets_u = torch.max(pseudo_labels, dim=1)  # [B]

                    # Calcola il max prob per ogni classe presente nel batch
                    for c in range(n_classes):
                        class_mask = (targets_u == c)
                        if class_mask.any():
                            max_c = max_probs[class_mask].max()
                            #mean_c = max_probs[class_mask].mean()
                            THRS[c] = min(ALPHA_FM * THRS[c] + (1 - ALPHA_FM) * max_c, TH_FIXMATCH) # clip at 0.95
                            #THRS[c] = min(ALPHA_FM * THRS[c] + (1 - ALPHA_FM) * mean_c, TH_FIXMATCH) # clip at 0.95

                    selected_thrs = THRS[targets_u]
                    mask = max_probs.ge(selected_thrs).float()

                u_pred_loss = (F.cross_entropy(pred_unl_target_strong, targets_u, reduction="none") * mask).mean()

                pred_unl_dom = torch.cat([pred_unl_target_strong_dom,pred_unl_target_dom],dim=0)
                u_loss_dom = loss_fn(pred_unl_dom, torch.ones(pred_unl_dom.shape[0]).long().to(device))

                inv_emb = torch.cat([emb_source_inv, emb_target_inv])
                spec_emb = torch.cat([emb_source_spec, emb_target_spec])
                unl_inv = torch.cat([unl_target_inv, unl_target_aug_inv],dim=0)
                unl_spec = torch.cat([unl_target_spec, unl_target_aug_spec],dim=0)

                if backbone == "ViT":
                    inv_emb = torch.abs(inv_emb)
                    spec_emb = torch.abs(spec_emb)
                    unl_inv = torch.abs(unl_inv)
                    unl_spec = torch.abs(unl_spec)

                norm_inv_emb = F.normalize(inv_emb)
                norm_spec_emb = F.normalize(spec_emb)
                norm_unl_inv = F.normalize(unl_inv)
                norm_unl_spec = F.normalize(unl_spec)

                loss_ortho = torch.mean(torch.sum( norm_inv_emb * norm_spec_emb, dim=1))
                u_loss_ortho = torch.mean( torch.sum( norm_unl_inv * norm_unl_spec, dim=1) )

                emb_t_all = torch.cat((emb_target_inv, unl_target_aug_inv), dim=0)  # all target embeddings (labelled + unlabelled)
                pred_t_all = torch.cat((task_target_cl, pred_unl_target_strong), dim=0)  # all target embeddings (labelled + unlabelled)

                ####################### GRL #######################

                sm_out_source = torch.softmax(task_source_cl.detach(), dim=1)
                sm_out_target = torch.softmax(pred_t_all.detach(), dim=1)

                cdan_features_source = torch.bmm(sm_out_source.unsqueeze(2), emb_source_inv.unsqueeze(1))
                cdan_features_target = torch.bmm(sm_out_target.unsqueeze(2), emb_t_all.unsqueeze(1))

                cdan_features_source = cdan_features_source.view(-1, sm_out_source.size(1) * emb_source_inv.size(1))
                cdan_features_target = cdan_features_target.view(-1, sm_out_target.size(1) * emb_t_all.size(1))

                dom_source_cl_grl = domain_cl(grl(cdan_features_source))
                dom_target_cl_grl = domain_cl(grl(cdan_features_target))

                pred_dom_grl = torch.cat([dom_source_cl_grl, dom_target_cl_grl], 0)
                y_batch_dom_grl = torch.cat([torch.zeros(TR_BATCH_SIZE, dtype=torch.long, device=device),
                                             torch.ones(TR_BATCH_SIZE, dtype=torch.long, device=device)], dim=0)

                loss_cl_pred = loss_fn(pred_dom_grl, y_batch_dom_grl)

                ####################### GRL #######################

                loss = loss_pred + loss_dom + u_loss_dom + loss_ortho + u_loss_ortho + u_pred_loss + loss_cl_pred

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot_loss+= loss.cpu().detach().numpy()
            tot_ortho_loss+=loss_ortho.cpu().detach().numpy()
            tot_fixmatch_loss+=u_pred_loss.cpu().detach().numpy()
            den+=1.

            torch.cuda.empty_cache()

        end = time.time()
        with autocast(device):
            pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
        f1_val = f1_score(labels_valid, pred_valid, average="weighted")
        
        ########################## EMA ##################################
        f1_val_ema = 0
        if epoch >= WARM_UP_EPOCH_EMA:
            ema_weights = cumulate_EMA(model, ema_weights, MOMENTUM_EMA)
            current_state_dict = model.state_dict()
            model.load_state_dict(ema_weights)
            with autocast(device):
                pred_valid, labels_valid = evaluation(model, dataloader_test_target, device)
            f1_val_ema = f1_score(labels_valid, pred_valid, average="weighted")
            f1_val_nw = f1_score(labels_valid, pred_valid, average=None)
            model.load_state_dict(current_state_dict)
        ########################## EMA ##################################
        
        pbar.set_postfix(
            {'Loss': tot_loss/den, 'F1 (ORIG)': 100*f1_val, 'F1 (EMA)': 100*f1_val_ema, 'Time': (end-start)})
        sys.stdout.flush()

    model_dir = os.path.join(out_dir, "models", source_prefix)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    output_file = os.path.join( model_dir, f"{source_prefix}_{nsplit}_{nsamples}.pth" )
    model.load_state_dict(ema_weights)
    torch.save(model.state_dict(), output_file)

    return 100 * f1_val_ema, 100 * f1_val_nw
