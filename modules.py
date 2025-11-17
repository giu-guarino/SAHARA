import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50, resnet34
import numpy as np
from functions import revgrad
from torch.autograd import Function

############## FROM GITHUB ####################

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)

#alpha = torch.tensor([1.])



class FC_Classifier_NoLazy_GRL(torch.nn.Module):
    def __init__(self, input_dim, n_classes, mid_channel=100, alpha=1., act=nn.ReLU()):
        super(FC_Classifier_NoLazy_GRL, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, mid_channel),
            act,
            nn.Linear(mid_channel, n_classes)
        )

        self.grl = GradientReversal(alpha=alpha)

    def forward(self, X):
        X_grl = self.grl(X)
        X_b = self.block(X_grl)
        return X_b

class FC_Classifier(torch.nn.Module):
    def __init__(self, input_dim, n_classes, mid_channel=100, act=nn.ReLU()):
        super(FC_Classifier, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, mid_channel),
            act,
            nn.Linear(mid_channel, n_classes)
        )

    def forward(self, X):
        X_b = self.block(X)
        return X_b

class FC_Classifier_2L(torch.nn.Module):
    def __init__(self, input_dim, n_classes, mid_channel=100, act=nn.ReLU()):
        super(FC_Classifier_2L, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, mid_channel),
            act,
            nn.Linear(mid_channel, mid_channel),
            act,
            nn.Linear(mid_channel, n_classes)
        )

    def forward(self, X):
        X_b = self.block(X)
        return X_b

class Proto_Classifier_GRL(torch.nn.Module):
    def __init__(self, input_dim, n_classes, alpha=1.):
        super(Proto_Classifier_GRL, self).__init__()

        self.block = ClassPrototypes(n_classes, input_dim)
        self.grl = GradientReversal(alpha=alpha)

    def forward(self, X):
        X_grl = self.grl(X)
        return self.block(X_grl)


class FC_Classifier_NoLazy(torch.nn.Module):
    def __init__(self, input_dim, n_classes):
        super(FC_Classifier_NoLazy, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, n_classes)
        )
    
    def forward(self, X):
        return self.block(X)

class ClassPrototypes(nn.Module):
    def __init__(self, n_classes, emb_size):
        super(ClassPrototypes, self).__init__()
        self.weight = nn.Parameter(torch.randn(emb_size, n_classes))
        self.LSM = nn.LogSoftmax(dim=1)
        #self.logsm = nn.Softmax(dim=1)

    def forward(self, x):
        # Use normalized weight during forward pass
        self._normalize_weight()  # Ensure self.weight is normalized
        prod = torch.matmul(x, self.weight)
        #print(f"prod = {prod}")
        return self.LSM(prod)

    def _normalize_weight(self):
        # Normalize weight in-place
        with torch.no_grad():
            self.weight.data = F.normalize(self.weight.data, p=2, dim=0)

    def get_weight(self):
        # Return the normalized weight
        self._normalize_weight()  # Ensure weight is normalized before returning
        return self.weight.detach().clone()

    def get_normalized_weight(self):
        # Explicitly return normalized weight for use in loss functions
        return F.normalize(self.weight, p=2, dim=0)

class ClassPrototypes2(nn.Module):
    def __init__(self, n_classes, emb_size):
        super(ClassPrototypes2, self).__init__()
        self.weight = nn.Parameter(torch.randn(emb_size, n_classes))

    def forward(self, x):
        # Use normalized weight during forward pass
        self._normalize_weight()  # Ensure self.weight is normalized
        prod = torch.matmul(x, self.weight)
        return prod

    def _normalize_weight(self):
        # Normalize weight in-place
        self.weight.data = F.normalize(self.weight.data, p=2, dim=0)

    def get_weight(self):
        # Return the normalized weight
        self._normalize_weight()  # Ensure weight is normalized before returning
        return self.weight.detach().clone()

    def get_normalized_weight(self):
        # Explicitly return normalized weight for use in loss functions
        return F.normalize(self.weight, p=2, dim=0)


class MultiPrototypeClassifier(nn.Module):
    def __init__(self, n_classes, emb_size, n_prototypes=3):
        super(MultiPrototypeClassifier, self).__init__()
        self.n_classes = n_classes
        self.n_prototypes = n_prototypes
        self.emb_size = emb_size

        # Peso con shape (emb_dim, n_classes * K)
        self.weight = nn.Parameter(torch.randn(emb_size, n_classes * n_prototypes))
        self._normalize_weight()

    def forward(self, x):
        # x: [B, emb_dim]
        self._normalize_weight()
        # Output: [B, n_classes * K]
        logits_all = torch.matmul(x, self.weight)

        # Reshape per classe: [B, n_classes, K]
        logits_per_class = logits_all.view(-1, self.n_classes, self.n_prototypes)

        # Aggregazione: max o mean sui prototipi
        logits = torch.max(logits_per_class, dim=2).values  # oppure .mean(dim=2)
        return logits

    def _normalize_weight(self):
        with torch.no_grad():
            self.weight.data = F.normalize(self.weight.data, p=2, dim=0)

    def get_normalized_weight(self):
        return F.normalize(self.weight, p=2, dim=0).view(self.emb_size, self.n_classes, self.n_prototypes)

    def get_prototypes_per_class(self):
        """Restituisce i prototipi normalizzati in forma: [n_classes, emb_size, n_prototypes]"""
        weight_norm = F.normalize(self.weight, p=2, dim=0)
        weight_norm = weight_norm.view(self.emb_size, self.n_classes, self.n_prototypes)
        return weight_norm.permute(1, 0, 2)  # [n_classes, emb_size, n_prototypes]


