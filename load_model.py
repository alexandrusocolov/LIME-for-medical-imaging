"""
Defines classes and functions to load CheXNet pre-trained model from https://github.com/arnoweng/CheXNet/blob/master/model.py
To use, just run load_model(CKPT_PATH='./model/model.pth.tar') where CKPT_PATH = path to where the model weights are.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from read_data import ChestXrayDataSet   #uncomment if you want to use ChestXRay-18 data
from sklearn.metrics import roc_auc_score
import pandas as pd
import cv2
from torch.utils.data import Dataset
from PIL import Image

N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, subset_img_ids, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            subset_img_ids: a subset of images that we actually have
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                if image_name in subset_img_ids:
                    label = items[1:]
                    label = [int(i) for i in label]
                    image_name = os.path.join(data_dir, image_name)
                    image_names.append(image_name)
                    labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

def compute_AUCs_modified(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    for i in range(N_CLASSES):
        if len(np.unique(gt[:, i])) == 1:
            AUROCs.append('AUC Not Defined')
        else:
            AUROCs.append(roc_auc_score(gt[:, i], pred[:, i]))
    return AUROCs


def clean_checkpoint(checkpoint):
    ''' remove "." between "norm"/"conv" and numbers in each key
        example: "module.densenet121.features.transition1.norm.1.weight" has to be
                 "module.densenet121.features.transition1.norm1.weight"
    '''

    keys_dont_touch = ["module.densenet121.features.transition1.norm.weight",
                       "module.densenet121.features.transition1.norm.bias",
                       "module.densenet121.features.transition1.norm.running_mean",
                       "module.densenet121.features.transition1.norm.running_var",
                       "module.densenet121.features.transition1.conv.weight",
                       "module.densenet121.features.transition2.norm.weight",
                       "module.densenet121.features.transition2.norm.bias",
                       "module.densenet121.features.transition2.norm.running_mean",
                       "module.densenet121.features.transition2.norm.running_var",
                       "module.densenet121.features.transition2.conv.weight",
                       "module.densenet121.features.transition3.norm.weight",
                       "module.densenet121.features.transition3.norm.bias",
                       "module.densenet121.features.transition3.norm.running_mean",
                       "module.densenet121.features.transition3.norm.running_var",
                       "module.densenet121.features.transition3.conv.weight"]

    checkpoint_adj = copy.deepcopy(checkpoint)
    for old_key in checkpoint['state_dict'].keys():
        if old_key not in keys_dont_touch:
            new_key = old_key.replace("norm.", "norm")
            checkpoint_adj['state_dict'][new_key] = checkpoint_adj['state_dict'].pop(old_key)

    checkpoint_adj_2 = copy.deepcopy(checkpoint_adj)
    for old_key in checkpoint_adj['state_dict'].keys():
        if old_key not in keys_dont_touch:
            new_key = old_key.replace("conv.", "conv")
            checkpoint_adj_2['state_dict'][new_key] = checkpoint_adj_2['state_dict'].pop(old_key)

    return checkpoint_adj_2


def load_model(CKPT_PATH='./model/model.pth.tar', N_CLASSES=14):
    # uncomment if you want to use ChestXRay-18 data
    # CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
    #                 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    # DATA_DIR = './ChestX-ray14/images'
    # TEST_IMAGE_LIST = './ChestX-ray14/labels/test_list.txt'
    # BATCH_SIZE = 64

    # initialize the model
    model = DenseNet121(N_CLASSES)
    model = torch.nn.DataParallel(model)

    # load the model
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH, map_location='cpu')

        checkpoint = clean_checkpoint(checkpoint)

        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    return model