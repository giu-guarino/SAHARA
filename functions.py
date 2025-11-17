from typing import Sequence
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms.functional as TF
from collections import OrderedDict
import torchvision.transforms as T
from torchvision.transforms import v2
from torch.autograd import Function

############## FROM GITHUB ####################
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - alpha * grad_output
        return grad_input, None

revgrad = GradientReversal.apply

################################################


class MyRotateTransform():
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)

        x = x.to(torch.float32)
        x = TF.rotate(x, angle)
        return x.to(torch.float16) if x.dtype == torch.float16 else x
        #return TF.rotate(x, angle)


angle = [0, 90, 180, 270]

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5),
    T.RandomApply([T.ColorJitter()], p=0.5)
    ])

transform_2 = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([MyRotateTransform(angles=angle)], p=0.5),
    #T.RandomApply([v2.GaussianNoise()], p=0.5)
    ])



def cumulate_EMA(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k].cpu().detach().numpy() + (1-alpha) * current_weights_npy[k]

    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )

    return current_weights

def cumulate_EMA_CL(model, projector, ema_weights, ema_weights_p, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    current_weights_p = OrderedDict()
    current_weights_p_npy = OrderedDict()

    state_dict = model.state_dict()
    state_dict_p = projector.state_dict()

    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    for k in state_dict_p:
        current_weights_p_npy[k] = state_dict_p[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k].cpu().detach().numpy() + (1-alpha) * current_weights_npy[k]

        for k in state_dict_p:
            current_weights_p_npy[k] = alpha * ema_weights_p[k].cpu().detach().numpy() + (1 - alpha) * \
                                     current_weights_p_npy[k]

    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )

    for k in state_dict_p:
        current_weights_p[k] = torch.tensor( current_weights_p_npy[k] )

    return current_weights, current_weights_p

def modify_weights(model, ema_weights, alpha):
    current_weights = OrderedDict()
    current_weights_npy = OrderedDict()
    state_dict = model.state_dict()
    
    for k in state_dict:
        current_weights_npy[k] = state_dict[k].cpu().detach().numpy()

    if ema_weights is not None:
        for k in state_dict:
            current_weights_npy[k] = alpha * ema_weights[k] + (1-alpha) * current_weights_npy[k]
    
    for k in state_dict:
        current_weights[k] = torch.tensor( current_weights_npy[k] )
    
    return current_weights, current_weights_npy


'''
class MyDataset_Unl(Dataset):
    def __init__(self, data, transform, data_idx=None):
        self.data = data
        self.transform = transform
        self.data_idx = data_idx
        
    def __getitem__(self, index):
        x = self.data[index]        
        x_transform = self.transform(self.data[index])
        if self.data_idx is not None:
            idx = self.data_idx[index]
        else:
            idx = -1
        
        return x, x_transform, idx
    
    def __len__(self):
        return len(self.data)
'''


class MyDataset_Unl(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        x_transform = self.transform(self.data[index])
        return x, x_transform

    def __len__(self):
        return len(self.data)



class MyDataset_Unl_idx(Dataset):
    def __init__(self, data, transform, idx):
        self.data = data
        self.transform = transform
        self.idx = idx

    def __getitem__(self, index):
        x = self.data[index]
        x_transform = self.transform(self.data[index])
        idxs = self.idx[index]

        return x, x_transform, idxs

    def __len__(self):
        return len(self.data)



class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
