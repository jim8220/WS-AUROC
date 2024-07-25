import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import matplotlib.pyplot as plt
import torch.backends.cudnn
import numpy as np
import os
import anomaly_dataset
import tqdm
from itertools import permutations
import itertools
import torch.nn.functional as F
import yaml
import joblib
import random
import utils
import pickle
import sklearn.mixture
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import copy

def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


# yaml file load

param = yaml_load()
torqueload_class = [0, 2, 4]

print(param)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')
if device == 'cuda':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


rseed = param['rseed']



set_randomseed = True
if set_randomseed:
    torch.manual_seed(rseed)
    torch.cuda.manual_seed(rseed)
    # torch.cuda.manual_seed_all(1) # Activate this line when you use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(rseed)
    torch.backends.cudnn.enabled = False
    random.seed(rseed)

# dataset generation
os.makedirs(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}', exist_ok=True)

# for fast
if param['short_cut'] and os.path.isfile(f'./dataset_prepared/train_data.npy'):
    print("train / valid / test dataset loaded from prepared dataset")

    train_data = np.load(f'./dataset_prepared/train_data.npy')
    train_npy_name = np.load(f'./dataset_prepared/train_npy_name.npy')

    valid_data = np.load(f'./dataset_prepared/valid_data.npy')
    valid_npy_name = np.load(f'./dataset_prepared/valid_npy_name.npy')

    test_data = np.load(f'./dataset_prepared/test_data.npy')
    test_npy_name = np.load(f'./dataset_prepared/test_npy_name.npy')

else:
    train_data, valid_data, test_data, train_npy_name, valid_npy_name, test_npy_name = anomaly_dataset.anomaly_dataset_extended(rseed=param['rseed'])

    train_npy_name = np.array(train_npy_name)
    valid_npy_name = np.array(valid_npy_name)
    test_npy_name = np.array(test_npy_name)


    np.save(f'./dataset_prepared/train_data', train_data)
    np.save(f'./dataset_prepared/train_npy_name', train_npy_name)

    np.save(f'./dataset_prepared/valid_data', valid_data)
    np.save(f'./dataset_prepared/valid_npy_name', valid_npy_name)

    np.save(f'./dataset_prepared/test_data', test_data)
    np.save(f'./dataset_prepared/test_npy_name', test_npy_name)

train_dataset = []

train_dataset = []

for inpy in range(len(train_npy_name)):

    torqueload = int(train_npy_name[inpy].split('/')[-1].split('_')[0][:-2])

    a_label = torqueload_class.index(torqueload)
    a_data = train_data[inpy, :, :]
    # append
    for jj in range(0, 4):
        train_dataset.append([a_data[:, jj], F.one_hot(torch.tensor(a_label), len(torqueload_class))])

train_dataset_copy = copy.deepcopy(train_dataset)
train_dataloader_for_test = DataLoader(train_dataset_copy, batch_size=1, shuffle=False)

# mixup

mixup_len = 3 * len(train_dataset)

for ii in range(mixup_len):
    lam = np.random.beta(1, 1)
    idx1 = np.random.randint(0, len(train_dataset))
    idx2 = np.random.randint(0, len(train_dataset))
    mix_npy = lam * train_dataset[idx1][0] + (1 - lam) * train_dataset[idx2][0]
    mix_label = lam * train_dataset[idx1][1] + (1 - lam) * train_dataset[idx2][1]
    train_dataset.append([mix_npy, mix_label])

# this is for valid dataset

valid_dataset = []

for inpy in range(len(valid_npy_name)):
    torqueload = int(valid_npy_name[inpy].split('/')[-1].split('_')[0][:-2])

    a_label = torqueload_class.index(torqueload)
    a_data = valid_data[inpy, :, :]
    # append
    for jj in range(0, 4):
        valid_dataset.append([a_data[:, jj], F.one_hot(torch.tensor(a_label), len(torqueload_class))])

# this is for train dataset

test_dataset = []

for inpy in range(len(test_npy_name)):
    a_data = test_data[inpy, :, :]
    a_name = test_npy_name[inpy]
    torqueload = int(test_npy_name[inpy].split('/')[-1].split('_')[0][:-2])
    a_label = torqueload_class.index(torqueload)
    a_anomaly = int('anomaly' in test_npy_name[inpy])
    # append
    test_dataset.append([a_data, a_label, a_anomaly, a_name])

# transfer dataset to dataloader

train_dataloader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)

valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model
detector = utils.vib_MFN_feat_shared_1000hz(len(torqueload_class)).to(device)

# loss function

optimizer = torch.optim.Adam(detector.parameters(), lr=1e-3)

best_model = detector
best_valid_loss = 10000000
val_losses = []

for t in range(param['epochs']):
    detector.train()
    print(f"Epoch {t + 1}\n-------------------------------")
    size = len(train_dataloader.dataset)
    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)

        pred = detector(X)
        target = y

        loss = torch.mean(-torch.log_softmax(pred, dim=1) * target)  # CCE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 30 == 0:
            print(f'train loss: {loss}')
    detector.eval()

    valid_loss = 0
    cnt = 0
    for batch, (X, y) in enumerate(valid_dataloader):
        with torch.no_grad():
            X = X.to(device)
            pred = detector(X).cpu().detach()
            y = y.cpu().detach()
            target = y

            valid_loss += torch.mean(-torch.log_softmax(pred, dim=1) * target)  # CCE
            cnt = cnt + 1
    valid_loss /= cnt

    print(f'valid loss: {valid_loss}')

    val_losses.append(valid_loss)
    if best_valid_loss >= valid_loss:  # threshold process should be added
        best_model = detector
        best_valid_loss = valid_loss

    print(f'best valid loss: {best_valid_loss}')

# test

# get train features

train_features = []
train_labels = []
for batch, (X, y) in enumerate(train_dataloader_for_test):
    with torch.no_grad():
        X = X.to(device)
        pred_feature = best_model.get_feature(X).squeeze(0).cpu().detach().numpy()
        train_features.append(pred_feature)
        train_labels.append(torch.argmax(y).item())

valid_features = []
valid_labels = []
for batch, (X, y) in enumerate(valid_dataloader):
    with torch.no_grad():
        X = X.to(device)
        pred_feature = best_model.get_feature(X).squeeze(0).cpu().detach().numpy()
        valid_features.append(pred_feature)
        valid_labels.append(torch.argmax(y).item())

# get test features

test_idxs = []
test_anomalies = []
test_names = []

test_features = []
for batch, (X, yidx, yanomaly, yname) in enumerate(test_dataloader):
    with torch.no_grad():
        X = X.to(device)

        test_idxs.append(yidx.item())
        test_anomalies.append(yanomaly.item())
        test_names.append(yname[0])

        for jj in range(0, 4):
            pred_feature = best_model.get_feature(X[:, :, jj]).squeeze(0).cpu().detach().numpy()
            test_features.append(pred_feature)

# define KNN model for anomaly detection for backend system

scaler_DL = StandardScaler()
train_features = scaler_DL.fit_transform(train_features)
test_features = scaler_DL.transform(test_features)

# knn = KNeighborsClassifier(n_neighbors=5)

knn_DL = KNeighborsClassifier(n_neighbors=1)
knn_DL.fit(train_features, train_labels)

test_predictions = knn_DL.predict(test_features)
test_pred_scores, _ = knn_DL.kneighbors(test_features)
test_pred_scores = [min(i) for i in test_pred_scores]

test_pred_sum_scores_DL = []

for idata in range(len(test_dataset)):
    test_pred_sum_scores_DL.append(sum(test_pred_scores[idata * 4:(idata + 1) * 4]))

valid_features = scaler_DL.transform(valid_features)
valid_predictions = knn_DL.predict(valid_features)
valid_pred_scores, _ = knn_DL.kneighbors(valid_features)
valid_pred_scores = [min(i) for i in valid_pred_scores]
valid_pred_sum_scores_DL = []
for idata in range(len(valid_dataset) // 4):
    valid_pred_sum_scores_DL.append(sum(valid_pred_scores[idata * 4:(idata + 1) * 4]))


if os.path.exists(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/train_fft.npy'):
    train_features = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/train_fft.npy')
    valid_features = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/valid_fft.npy')
    test_features = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/test_fft.npy')

    train_features_BPFI = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/train_fft_BPFI.npy')
    valid_features_BPFI = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/valid_fft_BPFI.npy')
    test_features_BPFI = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/test_fft_BPFI.npy')

    train_features_BPFO = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/train_fft_BPFO.npy')
    valid_features_BPFO = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/valid_fft_BPFO.npy')
    test_features_BPFO = np.load(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/test_fft_BPFO.npy')


else:

    train_features = []

    for ii in tqdm.tqdm(range(train_data.shape[0])):
        train_features.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(train_data)[ii,:,0]))[3010//60].item(),
                 torch.abs(torch.fft.fft(torch.tensor(train_data)[ii,:,1]))[3010//60].item(),
                 torch.abs(torch.fft.fft(torch.tensor(train_data)[ii,:,2]))[3010//60].item(),
                 torch.abs(torch.fft.fft(torch.tensor(train_data)[ii,:,3]))[3010//60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[4 * 3010 // 60].item()
                      ]))

    train_features = np.concatenate([np.expand_dims(tt, axis=0) for tt in train_features],axis=0)
    # valid

    valid_features = []

    for ii in tqdm.tqdm(range(valid_data.shape[0])):
        valid_features.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[2 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[2 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[2 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[2 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[4 * 3010 // 60].item()
                      ]))
    # test

    test_features = []

    for ii in tqdm.tqdm(range(test_data.shape[0])):
        test_features.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(test_data)[ii,:,0]))[3010//60].item(),
                 torch.abs(torch.fft.fft(torch.tensor(test_data)[ii,:,1]))[3010//60].item(),
                 torch.abs(torch.fft.fft(torch.tensor(test_data)[ii,:,2]))[3010//60].item(),
                 torch.abs(torch.fft.fft(torch.tensor(test_data)[ii,:,3]))[3010//60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[2*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[3*3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[4 * 3010 // 60].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[2 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[3 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[4 * 3010 // 60].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[4 * 3010 // 60].item()
                      ]))


    test_features = np.concatenate([np.expand_dims(tt, axis=0) for tt in test_features],axis=0)
    # get test features

    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/train_fft.npy',train_features)
    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/valid_fft.npy',valid_features)
    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/test_fft.npy',test_features)

    train_features_BPFI = []

    for ii in tqdm.tqdm(range(train_data.shape[0])):
        train_features_BPFI.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[4 * 272].item()
                      ]))

    train_features_BPFI = np.concatenate([np.expand_dims(tt, axis=0) for tt in train_features_BPFI], axis=0)
    # valid

    valid_features_BPFI = []

    for ii in tqdm.tqdm(range(valid_data.shape[0])):
        valid_features_BPFI.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[4 * 272].item()
                      ]))
    # test

    test_features_BPFI = []

    for ii in tqdm.tqdm(range(test_data.shape[0])):
        test_features_BPFI.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[2 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[3 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[4 * 272].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[2 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[3 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[4 * 272].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[4 * 272].item()
                      ]))

    test_features_BPFI = np.concatenate([np.expand_dims(tt, axis=0) for tt in test_features_BPFI], axis=0)
    # get test features

    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/train_fft_BPFI.npy', train_features_BPFI)
    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/valid_fft_BPFI.npy', valid_features_BPFI)
    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/test_fft_BPFI.npy', test_features_BPFI)

    train_features_BPFO = []

    for ii in tqdm.tqdm(range(train_data.shape[0])):
        train_features_BPFO.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 0]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 1]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 2]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(train_data)[ii, :, 3]))[4 * 179].item()
                      ]))

    train_features_BPFO = np.concatenate([np.expand_dims(tt, axis=0) for tt in train_features_BPFO], axis=0)
    # valid

    valid_features_BPFO = []

    for ii in tqdm.tqdm(range(valid_data.shape[0])):
        valid_features_BPFO.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 0]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 1]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 2]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(valid_data)[ii, :, 3]))[4 * 179].item()
                      ]))
    # test

    test_features_BPFO = []

    for ii in tqdm.tqdm(range(test_data.shape[0])):
        test_features_BPFO.append(
            np.array([torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[2 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[3 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[4 * 179].item(),
                      torch.abs(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[2 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[3 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 0]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 1]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 2]))[4 * 179].item(),
                      torch.angle(torch.fft.fft(torch.tensor(test_data)[ii, :, 3]))[4 * 179].item()
                      ]))

    test_features_BPFO = np.concatenate([np.expand_dims(tt, axis=0) for tt in test_features_BPFO], axis=0)
    # get test features

    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/train_fft_BPFO.npy', train_features_BPFO)
    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/valid_fft_BPFO.npy', valid_features_BPFO)
    np.save(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2/test_fft_BPFO.npy', test_features_BPFO)

test_anomalies = [int('anomaly' in kk) for kk in test_npy_name]
test_names = test_npy_name

train_labels = [torqueload_class.index(int(kk.split('/')[-1].split('_')[0][0])) for kk in train_npy_name]

train_features_ = np.concatenate(train_features,axis=0)

# define KNN model for anomaly detection for backend system

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)
valid_features = scaler.transform(valid_features)
#knn = KNeighborsClassifier(n_neighbors=5)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_features, train_labels)

test_predictions = knn.predict(test_features)
test_pred_scores, _ = knn.kneighbors(test_features)
test_pred_scores = [min(i) for i in test_pred_scores]


valid_pred_scores, _ = knn.kneighbors(valid_features)
valid_pred_scores = [min(i) for i in valid_pred_scores]


test_pred_sum_scores = test_pred_scores

valid_pred_sum_scores = valid_pred_scores

counts, bin_edges = np.histogram(valid_pred_sum_scores, bins=10, density=True)
cdf = np.cumsum(counts)
cdf = cdf / cdf[-1]
percentile_90_bin_index = np.where(cdf >= 0.90)[0][0]
percentile_90 = bin_edges[percentile_90_bin_index]

mu_sigma_3 = np.mean(valid_pred_sum_scores) + np.std(valid_pred_sum_scores) * 3


scaler_BPFI = StandardScaler()
train_features_BPFI = scaler_BPFI.fit_transform(train_features_BPFI)
test_features_BPFI = scaler_BPFI.transform(test_features_BPFI)
valid_features_BPFI = scaler_BPFI.transform(valid_features_BPFI)
#knn = KNeighborsClassifier(n_neighbors=5)

knn_BPFI = KNeighborsClassifier(n_neighbors=1)
knn_BPFI.fit(train_features_BPFI, train_labels)

test_predictions_BPFI = knn_BPFI.predict(test_features_BPFI)
test_pred_scores_BPFI, _ = knn_BPFI.kneighbors(test_features_BPFI)
test_pred_scores_BPFI = [min(i) for i in test_pred_scores_BPFI]


valid_pred_scores_BPFI, _ = knn_BPFI.kneighbors(valid_features_BPFI)
valid_pred_scores_BPFI = [min(i) for i in valid_pred_scores_BPFI]


test_pred_sum_scores_BPFI = test_pred_scores_BPFI

valid_pred_sum_scores_BPFI = valid_pred_scores_BPFI

counts_BPFI, bin_edges_BPFI = np.histogram(valid_pred_sum_scores_BPFI, bins=10, density=True)
cdf_BPFI = np.cumsum(counts_BPFI)
cdf_BPFI = cdf_BPFI / cdf_BPFI[-1]
percentile_90_bin_index_BPFI = np.where(cdf_BPFI >= 0.90)[0][0]
percentile_90_BPFI = bin_edges[percentile_90_bin_index_BPFI]

mu_sigma_3_BPFI = np.mean(valid_pred_sum_scores_BPFI) + np.std(valid_pred_sum_scores_BPFI) * 3

scaler_BPFO = StandardScaler()
train_features_BPFO = scaler_BPFO.fit_transform(train_features_BPFO)
test_features_BPFO = scaler_BPFO.transform(test_features_BPFO)
valid_features_BPFO = scaler_BPFO.transform(valid_features_BPFO)
#knn = KNeighborsClassifier(n_neighbors=5)

knn_BPFO = KNeighborsClassifier(n_neighbors=1)
knn_BPFO.fit(train_features_BPFO, train_labels)

test_predictions_BPFO = knn_BPFO.predict(test_features_BPFO)
test_pred_scores_BPFO, _ = knn_BPFO.kneighbors(test_features_BPFO)
test_pred_scores_BPFO = [min(i) for i in test_pred_scores_BPFO]


valid_pred_scores_BPFO, _ = knn.kneighbors(valid_features_BPFO)
valid_pred_scores_BPFO = [min(i) for i in valid_pred_scores_BPFO]


test_pred_sum_scores_BPFO = test_pred_scores_BPFO

valid_pred_sum_scores_BPFO = valid_pred_scores_BPFO

counts_BPFO, bin_edges_BPFO = np.histogram(valid_pred_sum_scores_BPFO, bins=10, density=True)
cdf_BPFO = np.cumsum(counts_BPFO)
cdf_BPFO = cdf_BPFO / cdf_BPFO[-1]
percentile_90_bin_index_BPFO = np.where(cdf_BPFO >= 0.90)[0][0]
percentile_90_BPFO = bin_edges[percentile_90_bin_index_BPFO]

mu_sigma_3_BPFO = np.mean(valid_pred_sum_scores_BPFO) + np.std(valid_pred_sum_scores_BPFO) * 3


S_ot = (np.array(test_pred_sum_scores)-np.min(valid_pred_sum_scores))/(np.max(valid_pred_sum_scores)-np.min(valid_pred_sum_scores))

S_BPFI = (np.array(test_pred_sum_scores_BPFI)-np.min(valid_pred_sum_scores_BPFI))/(np.max(valid_pred_sum_scores_BPFI)-np.min(valid_pred_sum_scores_BPFI))

S_BPFO = (np.array(test_pred_sum_scores_BPFO)-np.min(valid_pred_sum_scores_BPFO))/(np.max(valid_pred_sum_scores_BPFO)-np.min(valid_pred_sum_scores_BPFO))


#lambda_BPFIBPFO = np.tanh(S_BPFIBPFO)
#A_ot = np.exp(-1*np.array(test_pred_sum_scores))
#A_BPFIBPFO = np.exp(-1*np.array(test_pred_sum_scores_BPFIBPFO))


#HI = lambda_BPFIBPFO*S_BPFIBPFO + (1-lambda_BPFIBPFO)*S_ot

#HI = np.max([S_ot, S_BPFIBPFO],axis=0)

S_ot_valid = (np.array(valid_pred_sum_scores)-np.min(valid_pred_sum_scores))/(np.max(valid_pred_sum_scores)-np.min(valid_pred_sum_scores))

S_BPFI_valid = (np.array(valid_pred_sum_scores_BPFI)-np.min(valid_pred_sum_scores_BPFI))/(np.max(valid_pred_sum_scores_BPFI)-np.min(valid_pred_sum_scores_BPFI))
S_BPFO_valid = (np.array(valid_pred_sum_scores_BPFO)-np.min(valid_pred_sum_scores_BPFO))/(np.max(valid_pred_sum_scores_BPFO)-np.min(valid_pred_sum_scores_BPFO))

S_DL = (np.array(test_pred_sum_scores_DL)-np.min(valid_pred_sum_scores_DL))/(np.max(valid_pred_sum_scores_DL) - np.min(valid_pred_sum_scores_DL))

S_DL_valid = (np.array(valid_pred_sum_scores_DL)-np.min(valid_pred_sum_scores_DL))/(np.max(valid_pred_sum_scores_DL) - np.min(valid_pred_sum_scores_DL))


#HI = S_ot + S_BPFI + S_BPFO + S_DL
#HI_valid = S_ot_valid + S_BPFI_valid + S_BPFO_valid + S_DL_valid

HI = S_ot + S_BPFO + S_DL
HI_valid = S_ot_valid + S_BPFO_valid + S_DL_valid

mu_sigma_3_HI = np.mean(HI_valid) + 3 * np.std(HI_valid)

anomaly_type = ['Unbalance', 'BPFI','BPFO', 'Misalignment']
alen = int(len(test_names)/(len(anomaly_type)+1))
icount = 0
for ianomaly_type in anomaly_type:
    icount += 1

    itest_anomalies = test_anomalies[:alen] + test_anomalies[alen * icount:alen * (icount + 1)]
    iHI = HI[:alen].tolist() + HI[alen * icount:alen * (icount + 1)].tolist()
    ivtest_anomalies_HI = np.array(iHI) >= mu_sigma_3_HI

    fpr_HI, tpr_HI, thresholds_HI = sklearn.metrics.roc_curve(itest_anomalies, iHI)
    test_auc_HI = sklearn.metrics.auc(fpr_HI, tpr_HI)
    print(f'test auc (HI): {test_auc_HI}')
    f1_valthres_HI = sklearn.metrics.f1_score(itest_anomalies, ivtest_anomalies_HI, average='macro')
    print(f'valid thres f1 (HI) (mu+3sigma): {f1_valthres_HI}')

    # saving results

    f1s_HI = []
    thres_HI = []
    for ithres in np.linspace(min(HI), max(HI),1000):
        iitest_anomalies = np.array(iHI) >= ithres
        f1s_HI.append(sklearn.metrics.f1_score(itest_anomalies, iitest_anomalies, average='macro'))
        thres_HI.append(ithres)
    print(f'optimal f1 (HI): {max(f1s_HI)}')


    utils.plot_confusion_matrix_ad(itest_anomalies, np.array(np.array(iHI) >= thres_HI[np.argmax(f1s_HI)]),
                                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/confusion_matrix_anomaly_{ianomaly_type}.jpg',
                                title=f'confusion matrix {ianomaly_type}')

    plt.close()



    os.makedirs(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}', exist_ok=True)

    with open(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_aucs_HI_ASD_{ianomaly_type}.txt', 'w') as f_test:
        f_test.write(str(test_auc_HI))
        f_test.close()

    with open(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_aucs_HI_f1s_{ianomaly_type}.txt', 'w') as f_test:
        f_test.write(str(max(f1s_HI)))
        f_test.close()

    if ianomaly_type == 'Unbalance':

        mass_indexes_HI = [[],[],[],[],[]]
        mass_list = ['0583mg', '1169mg', '1751mg', '2239mg', '3318mg']
        masses = []
        for idata in range(len(test_names)):
            if 'Unbalance' in test_names[idata]:
                mass = test_names[idata].split('/')[-1].split('_')[2]
                mass_label = mass_list.index(mass)
                masses.append(mass_label)

                mass_indexes_HI[mass_label].append(HI[idata])
        # Create subplots
        fig, axs = plt.subplots(6, 1, sharex=True, figsize=(10, 20))

        axs[0].hist(HI[np.array(test_anomalies) == 0].tolist(), bins=50,
                    range=(min(iHI), max(iHI)), alpha=0.8, color='blue')
        #axs[0].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
        #axs[0].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
        axs[0].set_title('Normal', fontsize=24)
        axs[0].set_ylabel('Count', fontsize=24)
        #axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
        #axs[0].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}','normal'], loc='upper right')
        axs[0].set_ylim([0, 60])
        axs[0].tick_params(axis='both', which='major', labelsize=24)
        # Plot histograms for each mass index
        for i in range(5):
            axs[i + 1].hist(mass_indexes_HI[i], bins=50, range=(min(iHI), max(iHI)),
                            alpha=0.8, color='orange')
            #axs[i+1].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
            #axs[i+1].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
            axs[i + 1].set_title(f'Mass: {int(mass_list[i][:-2])} mg', fontsize=24)
            axs[i + 1].set_ylabel('Count', fontsize=24)
            #axs[i+1].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            # axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            #axs[i+1].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}',mass_list[i]], loc='upper right')
            axs[i+1].set_ylim([0, 20])
            axs[i+1].tick_params(axis='both', which='major', labelsize=24)
        # Set common labels
        plt.xlabel('Anomaly score', fontsize=24)
        plt.xlim([min(iHI), max(iHI)])
        plt.xticks(fontsize=24)  # Set fontsize for x-tick labels
        plt.yticks(fontsize=24)  # Set fontsize for y-tick labels

        plt.suptitle(f'anomaly={ianomaly_type} (AUROC:{np.round(test_auc_HI, 3)})', fontsize=32)

        # Save and display plot
        plt.savefig(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/score_ASD_{ianomaly_type}_detail_HI.svg', dpi=2000)
        plt.close()

    ###### proposed method ######
        normal_and_anomaly = [HI[:alen].tolist()] + mass_indexes_HI
        aucs = []
        penalty_physics = []
        penalty_uniform = []
        penalty = []
        combs = []
        mass_float = [float(imass[:-2]) for imass in mass_list]
        mass_float = [0.0] + mass_float
        mass_float_sum = 0
        for kk in itertools.combinations(range(len(normal_and_anomaly)),2):
            fpr_HI, tpr_HI, thresholds_HI = sklearn.metrics.roc_curve([0]*len(normal_and_anomaly[kk[0]])+[1]*len(normal_and_anomaly[kk[1]]), normal_and_anomaly[kk[0]]+normal_and_anomaly[kk[1]])
            test_auc_HI = sklearn.metrics.auc(fpr_HI, tpr_HI)
            aucs.append(test_auc_HI)
            dd = np.abs(kk[0]-kk[1])
            mass_dd = np.abs(mass_float[kk[0]]-mass_float[kk[1]])
            penalty.append(6*dd/((len(normal_and_anomaly)-1)*len(normal_and_anomaly)*(len(normal_and_anomaly)+1)))
            penalty_uniform.append(2/(len(normal_and_anomaly)*(len(normal_and_anomaly)-1)))
            penalty_physics.append(mass_dd)
            combs.append(kk)

        penalty_physics = np.array(penalty_physics)
        penalty_physics /= np.sum(penalty_physics)
        penalty_physics = penalty_physics.tolist()

        penalty_max = max(penalty_physics+penalty_uniform+penalty)

        minus = [penalty[qq]*(1-aucs[qq]) for qq in range(len(aucs))]
        minus_uniform = [penalty_uniform[qq]*(1-aucs[qq]) for qq in range(len(aucs))]
        minus_physics = [penalty_physics[qq] * (1 - aucs[qq]) for qq in range(len(aucs))]
        score = 1 - np.sum(minus)
        score_uniform = 1 - np.sum(minus_uniform)
        score_physics =  1 - np.sum(minus_physics)
        print(f'WS-AUROC (index) {ianomaly_type}: {np.round(score,3)}')
        print(f'WS-AUROC (uniform) {ianomaly_type}: {np.round(score_uniform,3)}')
        print(f'WS-AUROC (physics) {ianomaly_type}: {np.round(score_physics, 3)}')
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_index_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_uniform_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_uniform))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_physics_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_physics))
            f_test.close()


        # Create matrices for aucs and penalty
        matrix_size = len(normal_and_anomaly)
        auc_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix_uniform = np.zeros((matrix_size, matrix_size))
        penatly_matrix_physics = np.zeros((matrix_size, matrix_size))
        for idx, (i, j) in enumerate(combs):
            auc_matrix[i, j] = aucs[idx]
            auc_matrix[j, i] = aucs[idx]  # Symmetric matrix
            penalty_matrix[i, j] = penalty[idx]
            penalty_matrix[j, i] = penalty[idx]  # Symmetric matrix
            penalty_matrix_uniform[i, j] = penalty_uniform[idx]
            penalty_matrix_uniform[j, i] = penalty_uniform[idx]  # Symmetric matrix
            penatly_matrix_physics[i, j] = penalty_physics[idx]
            penatly_matrix_physics[j, i] = penalty_physics[idx]  # Symmetric matrix

        # Plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

        # Subplot 1: AUROC
        cax1 = ax1.matshow(auc_matrix, cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title('AUROC', fontsize=32)
        ax1.set_xlabel('Severity level')
        ax1.set_ylabel('Severity level')

        # Annotate the AUC matrix with values
        for (i, j), val in np.ndenumerate(auc_matrix):
            color = 'white' if auc_matrix[i, j] < 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 2: Penalty (uniform)
        cax2 = ax2.matshow(penalty_matrix_uniform, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax2, ax=ax2)
        ax2.set_title('Penalty (uniform)', fontsize=32)
        ax2.set_xlabel('Severity level')
        ax2.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix_uniform):
            color = 'white' if penalty_matrix_uniform[i, j] < 0.5 * np.max(penalty_matrix_uniform) else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 3: Penalty (index)
        cax3 = ax3.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax3, ax=ax3)
        ax3.set_title('Penalty (index)', fontsize=32)
        ax3.set_xlabel('Severity level')
        ax3.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix):
            color = 'white' if penalty_matrix[i, j] < 0.5 * np.max(penalty_matrix) else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)


        # Subplot 3: Penalty (physics)
        cax4 = ax4.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax4, ax=ax4)
        ax4.set_title('Penalty (physics)', fontsize=32)
        ax4.set_xlabel('Severity level')
        ax4.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penatly_matrix_physics):
            color = 'white' if penatly_matrix_physics[i, j] < 0.5 * np.max(penatly_matrix_physics) else 'black'
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Add xticks and yticks
        xticks_labels = ['normal'] + mass_list
        ax1.set_xticks(range(len(xticks_labels)))
        ax1.set_yticks(range(len(xticks_labels)))
        ax1.set_xticklabels(xticks_labels)
        ax1.set_yticklabels(xticks_labels)

        ax2.set_xticks(range(len(xticks_labels)))
        ax2.set_yticks(range(len(xticks_labels)))
        ax2.set_xticklabels(xticks_labels)
        ax2.set_yticklabels(xticks_labels)

        ax3.set_xticks(range(len(xticks_labels)))
        ax3.set_yticks(range(len(xticks_labels)))
        ax3.set_xticklabels(xticks_labels)
        ax3.set_yticklabels(xticks_labels)

        ax4.set_xticks(range(len(xticks_labels)))
        ax4.set_yticks(range(len(xticks_labels)))
        ax4.set_xticklabels(xticks_labels)
        ax4.set_yticklabels(xticks_labels)

        # Move xtick labels to the bottom
        ax1.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_ticks_position('bottom')
        ax4.xaxis.set_ticks_position('bottom')



        # Add a suptitle with the score and anomaly type
        plt.suptitle(f'anomaly={ianomaly_type}', fontsize=32)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/proposed_fig_{ianomaly_type}.svg', dpi=2000)
        plt.close()


    elif ianomaly_type == 'BPFI':

        crack_indexes_HI = [[], [], []]
        crack_list = ['03', '10', '30']
        cracks = []
        for idata in range(len(test_names)):
            if 'BPFI' in test_names[idata]:
                crack = test_names[idata].split('/')[-1].split('_')[2]
                crack_label = crack_list.index(crack)
                cracks.append(crack_label)

                crack_indexes_HI[crack_label].append(HI[idata])
        # Create subplots

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 20))

        axs[0].hist(HI[np.array(test_anomalies) == 0].tolist(), bins=50,
                    range=(min(iHI), max(iHI)), alpha=0.8, color='blue')
        #axs[0].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
        #axs[0].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
        axs[0].set_title('Normal', fontsize=24)
        axs[0].set_ylabel('Count', fontsize=24)
        #axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
        #axs[0].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}', 'normal'], loc='upper right')
        axs[0].set_ylim([0, 60])
        axs[0].tick_params(axis='both', which='major', labelsize=24)
        # Plot histograms for each mass index
        for i in range(3):
            axs[i + 1].hist(crack_indexes_HI[i], bins=50, range=(min(iHI), max(iHI)),
                            alpha=0.8, color='orange')
            #axs[i+1].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
            #axs[i+1].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
            axs[i + 1].set_title(f'Crack: {float(crack_list[i])/10} mm', fontsize=24)
            axs[i + 1].set_ylabel('Count', fontsize=24)
            #axs[i+1].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            # axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            #axs[i+1].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}', float(crack_list[i])/10], loc='upper right')
            axs[i+1].set_ylim([0, 20])
            axs[i+1].tick_params(axis='both', which='major', labelsize=24)
        # Set common labels
        plt.xlabel('Anomaly score', fontsize=24)
        plt.xlim([min(iHI), max(iHI)])
        plt.xticks(fontsize=24)  # Set fontsize for x-tick labels
        plt.yticks(fontsize=24)  # Set fontsize for y-tick labels

        plt.suptitle(f'anomaly={ianomaly_type} (AUROC:{np.round(test_auc_HI, 3)})', fontsize=32)

        # Save and display plot
        plt.savefig(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/score_ASD_{ianomaly_type}_detail_HI.svg', dpi=2000)
        plt.close()

    ###### proposed method ######
        normal_and_anomaly = [HI[:alen].tolist()] + crack_indexes_HI
        aucs = []
        penalty_physics = []
        penalty_uniform = []
        penalty = []
        combs = []
        crack_float = [float(icrack) for icrack in crack_list]
        crack_float = [0.0] + crack_float
        crack_float_sum = 0
        for kk in itertools.combinations(range(len(normal_and_anomaly)),2):
            fpr_HI, tpr_HI, thresholds_HI = sklearn.metrics.roc_curve([0]*len(normal_and_anomaly[kk[0]])+[1]*len(normal_and_anomaly[kk[1]]), normal_and_anomaly[kk[0]]+normal_and_anomaly[kk[1]])
            test_auc_HI = sklearn.metrics.auc(fpr_HI, tpr_HI)
            aucs.append(test_auc_HI)
            dd = np.abs(kk[0]-kk[1])
            crack_dd = np.abs(crack_float[kk[0]]-crack_float[kk[1]])
            penalty.append(6*dd/((len(normal_and_anomaly)-1)*len(normal_and_anomaly)*(len(normal_and_anomaly)+1)))
            penalty_uniform.append(2/(len(normal_and_anomaly)*(len(normal_and_anomaly)-1)))
            penalty_physics.append(crack_dd)
            combs.append(kk)

        penalty_physics = np.array(penalty_physics)
        penalty_physics /= np.sum(penalty_physics)
        penalty_physics = penalty_physics.tolist()

        penalty_max = max(penalty_physics+penalty_uniform+penalty)

        minus = [penalty[qq]*(1-aucs[qq]) for qq in range(len(aucs))]
        minus_uniform = [penalty_uniform[qq]*(1-aucs[qq]) for qq in range(len(aucs))]
        minus_physics = [penalty_physics[qq] * (1 - aucs[qq]) for qq in range(len(aucs))]
        score = 1 - np.sum(minus)
        score_uniform = 1 - np.sum(minus_uniform)
        score_physics =  1 - np.sum(minus_physics)
        print(f'WS-AUROC (index) {ianomaly_type}: {np.round(score,3)}')
        print(f'WS-AUROC (uniform) {ianomaly_type}: {np.round(score_uniform,3)}')
        print(f'WS-AUROC (physics) {ianomaly_type}: {np.round(score_physics, 3)}')
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_index_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_uniform_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_uniform))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_physics_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_physics))
            f_test.close()


        # Create matrices for aucs and penalty
        matrix_size = len(normal_and_anomaly)
        auc_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix_uniform = np.zeros((matrix_size, matrix_size))
        penatly_matrix_physics = np.zeros((matrix_size, matrix_size))
        for idx, (i, j) in enumerate(combs):
            auc_matrix[i, j] = aucs[idx]
            auc_matrix[j, i] = aucs[idx]  # Symmetric matrix
            penalty_matrix[i, j] = penalty[idx]
            penalty_matrix[j, i] = penalty[idx]  # Symmetric matrix
            penalty_matrix_uniform[i, j] = penalty_uniform[idx]
            penalty_matrix_uniform[j, i] = penalty_uniform[idx]  # Symmetric matrix
            penatly_matrix_physics[i, j] = penalty_physics[idx]
            penatly_matrix_physics[j, i] = penalty_physics[idx]  # Symmetric matrix

        # Plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

        # Subplot 1: AUROC
        cax1 = ax1.matshow(auc_matrix, cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title('AUROC', fontsize=32)
        ax1.set_xlabel('Severity level')
        ax1.set_ylabel('Severity level')

        # Annotate the AUC matrix with values
        for (i, j), val in np.ndenumerate(auc_matrix):
            color = 'white' if auc_matrix[i, j] < 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 2: Penalty (uniform)
        cax2 = ax2.matshow(penalty_matrix_uniform, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax2, ax=ax2)
        ax2.set_title('Penalty (uniform)', fontsize=32)
        ax2.set_xlabel('Severity level')
        ax2.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix_uniform):
            color = 'white' if penalty_matrix_uniform[i, j] < 0.5 * np.max(penalty_matrix_uniform) else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 3: Penalty (index)
        cax3 = ax3.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax3, ax=ax3)
        ax3.set_title('Penalty (index)', fontsize=32)
        ax3.set_xlabel('Severity level')
        ax3.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix):
            color = 'white' if penalty_matrix[i, j] < 0.5 * np.max(penalty_matrix) else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)


        # Subplot 3: Penalty (physics)
        cax4 = ax4.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax4, ax=ax4)
        ax4.set_title('Penalty (physics)', fontsize=32)
        ax4.set_xlabel('Severity level')
        ax4.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penatly_matrix_physics):
            color = 'white' if penatly_matrix_physics[i, j] < 0.5 * np.max(penatly_matrix_physics) else 'black'
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Add xticks and yticks
        xticks_labels = ['normal'] + crack_list
        ax1.set_xticks(range(len(xticks_labels)))
        ax1.set_yticks(range(len(xticks_labels)))
        ax1.set_xticklabels(xticks_labels)
        ax1.set_yticklabels(xticks_labels)

        ax2.set_xticks(range(len(xticks_labels)))
        ax2.set_yticks(range(len(xticks_labels)))
        ax2.set_xticklabels(xticks_labels)
        ax2.set_yticklabels(xticks_labels)

        ax3.set_xticks(range(len(xticks_labels)))
        ax3.set_yticks(range(len(xticks_labels)))
        ax3.set_xticklabels(xticks_labels)
        ax3.set_yticklabels(xticks_labels)

        ax4.set_xticks(range(len(xticks_labels)))
        ax4.set_yticks(range(len(xticks_labels)))
        ax4.set_xticklabels(xticks_labels)
        ax4.set_yticklabels(xticks_labels)

        # Move xtick labels to the bottom
        ax1.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_ticks_position('bottom')
        ax4.xaxis.set_ticks_position('bottom')



        # Add a suptitle with the score and anomaly type
        plt.suptitle(f'anomaly={ianomaly_type}', fontsize=32)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/proposed_fig_{ianomaly_type}.svg', dpi=2000)
        plt.close()

    elif ianomaly_type == 'BPFO':

        crack_indexes_HI = [[], [], []]
        crack_list = ['03', '10', '30']
        cracks = []
        for idata in range(len(test_names)):
            if 'BPFO' in test_names[idata]:
                crack = test_names[idata].split('/')[-1].split('_')[2]
                crack_label = crack_list.index(crack)
                cracks.append(crack_label)

                crack_indexes_HI[crack_label].append(HI[idata])
        # Create subplots

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 20))

        axs[0].hist(HI[np.array(test_anomalies) == 0].tolist(), bins=50,
                    range=(min(iHI), max(iHI)), alpha=0.8, color='blue')
        #axs[0].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
        #axs[0].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
        axs[0].set_title('Normal', fontsize=24)
        axs[0].set_ylabel('Count', fontsize=24)
        #axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
        #axs[0].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}', 'normal'], loc='upper right')
        axs[0].set_ylim([0, 60])
        axs[0].tick_params(axis='both', which='major', labelsize=24)
        # Plot histograms for each mass index
        for i in range(3):
            axs[i + 1].hist(crack_indexes_HI[i], bins=50, range=(min(iHI), max(iHI)),
                            alpha=0.8, color='orange')
            #axs[i+1].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
            #axs[i+1].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
            axs[i + 1].set_title(f'Crack: {float(crack_list[i])/10} mm', fontsize=24)
            axs[i + 1].set_ylabel('Count', fontsize=24)
            #axs[i+1].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            # axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            #axs[i+1].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}', float(crack_list[i])/10], loc='upper right')
            axs[i+1].set_ylim([0, 20])
            axs[i+1].tick_params(axis='both', which='major', labelsize=24)
        # Set common labels
        plt.xlabel('Anomaly score', fontsize=24)
        plt.xlim([min(iHI), max(iHI)])
        plt.xticks(fontsize=24)  # Set fontsize for x-tick labels
        plt.yticks(fontsize=24)  # Set fontsize for y-tick labels

        plt.suptitle(f'anomaly={ianomaly_type} (AUROC:{np.round(test_auc_HI, 3)})', fontsize=32)

        # Save and display plot
        plt.savefig(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/score_ASD_{ianomaly_type}_detail_HI.svg', dpi=2000)
        plt.close()


    ###### proposed method ######
        normal_and_anomaly = [HI[:alen].tolist()] + crack_indexes_HI
        aucs = []
        penalty_physics = []
        penalty_uniform = []
        penalty = []
        combs = []
        crack_float = [float(icrack) for icrack in crack_list]
        crack_float = [0.0] + crack_float
        crack_float_sum = 0
        for kk in itertools.combinations(range(len(normal_and_anomaly)),2):
            fpr_HI, tpr_HI, thresholds_HI = sklearn.metrics.roc_curve([0]*len(normal_and_anomaly[kk[0]])+[1]*len(normal_and_anomaly[kk[1]]), normal_and_anomaly[kk[0]]+normal_and_anomaly[kk[1]])
            test_auc_HI = sklearn.metrics.auc(fpr_HI, tpr_HI)
            aucs.append(test_auc_HI)
            dd = np.abs(kk[0]-kk[1])
            crack_dd = np.abs(crack_float[kk[0]]-crack_float[kk[1]])
            penalty.append(6*dd/((len(normal_and_anomaly)-1)*len(normal_and_anomaly)*(len(normal_and_anomaly)+1)))
            penalty_uniform.append(2/(len(normal_and_anomaly)*(len(normal_and_anomaly)-1)))
            penalty_physics.append(crack_dd)
            combs.append(kk)

        penalty_physics = np.array(penalty_physics)
        penalty_physics /= np.sum(penalty_physics)
        penalty_physics = penalty_physics.tolist()

        penalty_max = max(penalty_physics+penalty_uniform+penalty)

        minus = [penalty[qq]*(1-aucs[qq]) for qq in range(len(aucs))]
        minus_uniform = [penalty_uniform[qq]*(1-aucs[qq]) for qq in range(len(aucs))]
        minus_physics = [penalty_physics[qq] * (1 - aucs[qq]) for qq in range(len(aucs))]
        score = 1 - np.sum(minus)
        score_uniform = 1 - np.sum(minus_uniform)
        score_physics =  1 - np.sum(minus_physics)
        print(f'WS-AUROC (index) {ianomaly_type}: {np.round(score,3)}')
        print(f'WS-AUROC (uniform) {ianomaly_type}: {np.round(score_uniform,3)}')
        print(f'WS-AUROC (physics) {ianomaly_type}: {np.round(score_physics, 3)}')
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_index_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_uniform_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_uniform))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_physics_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_physics))
            f_test.close()


        # Create matrices for aucs and penalty
        matrix_size = len(normal_and_anomaly)
        auc_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix_uniform = np.zeros((matrix_size, matrix_size))
        penatly_matrix_physics = np.zeros((matrix_size, matrix_size))
        for idx, (i, j) in enumerate(combs):
            auc_matrix[i, j] = aucs[idx]
            auc_matrix[j, i] = aucs[idx]  # Symmetric matrix
            penalty_matrix[i, j] = penalty[idx]
            penalty_matrix[j, i] = penalty[idx]  # Symmetric matrix
            penalty_matrix_uniform[i, j] = penalty_uniform[idx]
            penalty_matrix_uniform[j, i] = penalty_uniform[idx]  # Symmetric matrix
            penatly_matrix_physics[i, j] = penalty_physics[idx]
            penatly_matrix_physics[j, i] = penalty_physics[idx]  # Symmetric matrix

        # Plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

        # Subplot 1: AUROC
        cax1 = ax1.matshow(auc_matrix, cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title('AUROC', fontsize=32)
        ax1.set_xlabel('Severity level')
        ax1.set_ylabel('Severity level')

        # Annotate the AUC matrix with values
        for (i, j), val in np.ndenumerate(auc_matrix):
            color = 'white' if auc_matrix[i, j] < 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 2: Penalty (uniform)
        cax2 = ax2.matshow(penalty_matrix_uniform, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax2, ax=ax2)
        ax2.set_title('Penalty (uniform)', fontsize=32)
        ax2.set_xlabel('Severity level')
        ax2.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix_uniform):
            color = 'white' if penalty_matrix_uniform[i, j] < 0.5 * np.max(penalty_matrix_uniform) else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 3: Penalty (index)
        cax3 = ax3.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax3, ax=ax3)
        ax3.set_title('Penalty (index)', fontsize=32)
        ax3.set_xlabel('Severity level')
        ax3.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix):
            color = 'white' if penalty_matrix[i, j] < 0.5 * np.max(penalty_matrix) else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)


        # Subplot 3: Penalty (physics)
        cax4 = ax4.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax4, ax=ax4)
        ax4.set_title('Penalty (physics)', fontsize=32)
        ax4.set_xlabel('Severity level')
        ax4.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penatly_matrix_physics):
            color = 'white' if penatly_matrix_physics[i, j] < 0.5 * np.max(penatly_matrix_physics) else 'black'
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Add xticks and yticks
        xticks_labels = ['normal'] + crack_list
        ax1.set_xticks(range(len(xticks_labels)))
        ax1.set_yticks(range(len(xticks_labels)))
        ax1.set_xticklabels(xticks_labels)
        ax1.set_yticklabels(xticks_labels)

        ax2.set_xticks(range(len(xticks_labels)))
        ax2.set_yticks(range(len(xticks_labels)))
        ax2.set_xticklabels(xticks_labels)
        ax2.set_yticklabels(xticks_labels)

        ax3.set_xticks(range(len(xticks_labels)))
        ax3.set_yticks(range(len(xticks_labels)))
        ax3.set_xticklabels(xticks_labels)
        ax3.set_yticklabels(xticks_labels)

        ax4.set_xticks(range(len(xticks_labels)))
        ax4.set_yticks(range(len(xticks_labels)))
        ax4.set_xticklabels(xticks_labels)
        ax4.set_yticklabels(xticks_labels)

        # Move xtick labels to the bottom
        ax1.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_ticks_position('bottom')
        ax4.xaxis.set_ticks_position('bottom')



        # Add a suptitle with the score and anomaly type
        plt.suptitle(f'anomaly={ianomaly_type}', fontsize=32)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/proposed_fig_{ianomaly_type}.svg', dpi=2000)
        plt.close()



    elif ianomaly_type == 'Misalignment':
        misalign_indexes = [[],[],[]]
        misalign_indexes_HI = [[], [], []]
        misalign_list = ['01', '03', '05']
        misaligns = []
        for idata in range(len(test_names)):
            if 'Misalign' in test_names[idata]:
                misalign = test_names[idata].split('/')[-1].split('_')[2]
                misalign_label = misalign_list.index(misalign)
                misaligns.append(misalign_label)
                misalign_indexes[misalign_label].append(test_pred_sum_scores[idata])
                misalign_indexes_HI[misalign_label].append(HI[idata])
        # Create subplots
        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 20))

        axs[0].hist(HI[np.array(test_anomalies) == 0].tolist(), bins=50,
                    range=(min(iHI), max(iHI)), alpha=0.8, color='blue')
        #axs[0].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
        #axs[0].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
        axs[0].set_title('Normal', fontsize=24)
        axs[0].set_ylabel('Count', fontsize=24)
        #axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
        #axs[0].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}','normal'], loc='upper right')
        axs[0].set_ylim([0, 60])
        axs[0].tick_params(axis='both', which='major', labelsize=24)
        # Plot histograms for each mass index
        for i in range(3):
            axs[i + 1].hist(misalign_indexes_HI[i], bins=50, range=(min(iHI), max(iHI)),
                            alpha=0.8, color='orange')
            #axs[i+1].axvline(x=np.round(thres[np.argmax(f1s)], 3), color='red', linewidth=2)
            #axs[i+1].axvline(x=np.round(mu_sigma_3_HI, 3), color='green', linewidth=2)
            axs[i + 1].set_title(f'Misalignment: {float(misalign_list[i])/10} mm', fontsize=24)
            axs[i + 1].set_ylabel('Count', fontsize=24)
            #axs[i+1].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            # axs[0].legend([f'optimal threshold: {np.round(thres[np.argmax(f1s)],3)}', f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_set)}', 'normal'],loc='upper right')
            #axs[i+1].legend([f'valid $\mu+3\sigma$ threshold: {np.round(mu_sigma_3_HI,3)}',float(misalign_list[i])/10], loc='upper right')
            axs[i+1].set_ylim([0, 20])
            axs[i+1].tick_params(axis='both', which='major', labelsize=24)
        # Set common labels
        plt.xlabel('Anomaly score', fontsize=24)
        plt.xlim([min(iHI), max(iHI)])
        plt.xticks(fontsize=24)  # Set fontsize for x-tick labels
        plt.yticks(fontsize=24)  # Set fontsize for y-tick labels

        plt.suptitle(f'anomaly={ianomaly_type} (AUROC:{np.round(test_auc_HI, 3)})', fontsize=32)

        # Save and display plot
        plt.savefig(f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/score_ASD_{ianomaly_type}_detail_HI.svg', dpi=2000)
        plt.close()

        ###### proposed method ######
        normal_and_anomaly = [HI[:alen].tolist()] + misalign_indexes_HI
        aucs = []
        penalty_physics = []
        penalty_uniform = []
        penalty = []
        combs = []
        misalign_float = [float(imisalign) for imisalign in misalign_list]
        misalign_float = [0.0] + misalign_float
        misalign_float_sum = 0
        for kk in itertools.combinations(range(len(normal_and_anomaly)), 2):
            fpr_HI, tpr_HI, thresholds_HI = sklearn.metrics.roc_curve(
                [0] * len(normal_and_anomaly[kk[0]]) + [1] * len(normal_and_anomaly[kk[1]]),
                normal_and_anomaly[kk[0]] + normal_and_anomaly[kk[1]])
            test_auc_HI = sklearn.metrics.auc(fpr_HI, tpr_HI)
            aucs.append(test_auc_HI)
            dd = np.abs(kk[0] - kk[1])
            misalign_dd = np.abs(misalign_float[kk[0]] - misalign_float[kk[1]])
            penalty.append(
                6 * dd / ((len(normal_and_anomaly) - 1) * len(normal_and_anomaly) * (len(normal_and_anomaly) + 1)))
            penalty_uniform.append(2 / (len(normal_and_anomaly) * (len(normal_and_anomaly) - 1)))
            penalty_physics.append(misalign_dd)
            combs.append(kk)

        penalty_physics = np.array(penalty_physics)
        penalty_physics /= np.sum(penalty_physics)
        penalty_physics = penalty_physics.tolist()

        penalty_max = max(penalty_physics + penalty_uniform + penalty)

        minus = [penalty[qq] * (1 - aucs[qq]) for qq in range(len(aucs))]
        minus_uniform = [penalty_uniform[qq] * (1 - aucs[qq]) for qq in range(len(aucs))]
        minus_physics = [penalty_physics[qq] * (1 - aucs[qq]) for qq in range(len(aucs))]
        score = 1 - np.sum(minus)
        score_uniform = 1 - np.sum(minus_uniform)
        score_physics = 1 - np.sum(minus_physics)
        print(f'WS-AUROC (index) {ianomaly_type}: {np.round(score, 3)}')
        print(f'WS-AUROC (uniform) {ianomaly_type}: {np.round(score_uniform, 3)}')
        print(f'WS-AUROC (physics) {ianomaly_type}: {np.round(score_physics, 3)}')
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_index_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_uniform_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_uniform))
            f_test.close()
        with open(
                f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/test_WS-AUROC_physics_HI_ASD_{ianomaly_type}.txt',
                'w') as f_test:
            f_test.write(str(score_physics))
            f_test.close()

        # Create matrices for aucs and penalty
        matrix_size = len(normal_and_anomaly)
        auc_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix = np.zeros((matrix_size, matrix_size))
        penalty_matrix_uniform = np.zeros((matrix_size, matrix_size))
        penatly_matrix_physics = np.zeros((matrix_size, matrix_size))
        for idx, (i, j) in enumerate(combs):
            auc_matrix[i, j] = aucs[idx]
            auc_matrix[j, i] = aucs[idx]  # Symmetric matrix
            penalty_matrix[i, j] = penalty[idx]
            penalty_matrix[j, i] = penalty[idx]  # Symmetric matrix
            penalty_matrix_uniform[i, j] = penalty_uniform[idx]
            penalty_matrix_uniform[j, i] = penalty_uniform[idx]  # Symmetric matrix
            penatly_matrix_physics[i, j] = penalty_physics[idx]
            penatly_matrix_physics[j, i] = penalty_physics[idx]  # Symmetric matrix

        # Plotting
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))

        # Subplot 1: AUROC
        cax1 = ax1.matshow(auc_matrix, cmap='viridis', vmin=0, vmax=1)
        fig.colorbar(cax1, ax=ax1)
        ax1.set_title('AUROC', fontsize=32)
        ax1.set_xlabel('Severity level')
        ax1.set_ylabel('Severity level')

        # Annotate the AUC matrix with values
        for (i, j), val in np.ndenumerate(auc_matrix):
            color = 'white' if auc_matrix[i, j] < 0.5 else 'black'
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 2: Penalty (uniform)
        cax2 = ax2.matshow(penalty_matrix_uniform, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax2, ax=ax2)
        ax2.set_title('Penalty (uniform)', fontsize=32)
        ax2.set_xlabel('Severity level')
        ax2.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix_uniform):
            color = 'white' if penalty_matrix_uniform[i, j] < 0.5 * np.max(penalty_matrix_uniform) else 'black'
            ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 3: Penalty (index)
        cax3 = ax3.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax3, ax=ax3)
        ax3.set_title('Penalty (index)', fontsize=32)
        ax3.set_xlabel('Severity level')
        ax3.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penalty_matrix):
            color = 'white' if penalty_matrix[i, j] < 0.5 * np.max(penalty_matrix) else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Subplot 3: Penalty (physics)
        cax4 = ax4.matshow(penalty_matrix, cmap='viridis', vmin=0, vmax=penalty_max)
        fig.colorbar(cax4, ax=ax4)
        ax4.set_title('Penalty (physics)', fontsize=32)
        ax4.set_xlabel('Severity level')
        ax4.set_ylabel('Severity level')

        # Annotate the Penalty with values
        for (i, j), val in np.ndenumerate(penatly_matrix_physics):
            color = 'white' if penatly_matrix_physics[i, j] < 0.5 * np.max(penatly_matrix_physics) else 'black'
            ax4.text(j, i, f'{val:.2f}', ha='center', va='center', color=color)

        # Add xticks and yticks
        xticks_labels = ['normal'] + misalign_list
        ax1.set_xticks(range(len(xticks_labels)))
        ax1.set_yticks(range(len(xticks_labels)))
        ax1.set_xticklabels(xticks_labels)
        ax1.set_yticklabels(xticks_labels)

        ax2.set_xticks(range(len(xticks_labels)))
        ax2.set_yticks(range(len(xticks_labels)))
        ax2.set_xticklabels(xticks_labels)
        ax2.set_yticklabels(xticks_labels)

        ax3.set_xticks(range(len(xticks_labels)))
        ax3.set_yticks(range(len(xticks_labels)))
        ax3.set_xticklabels(xticks_labels)
        ax3.set_yticklabels(xticks_labels)

        ax4.set_xticks(range(len(xticks_labels)))
        ax4.set_yticks(range(len(xticks_labels)))
        ax4.set_xticklabels(xticks_labels)
        ax4.set_yticklabels(xticks_labels)

        # Move xtick labels to the bottom
        ax1.xaxis.set_ticks_position('bottom')
        ax2.xaxis.set_ticks_position('bottom')
        ax3.xaxis.set_ticks_position('bottom')
        ax4.xaxis.set_ticks_position('bottom')

        # Add a suptitle with the score and anomaly type
        plt.suptitle(f'anomaly={ianomaly_type}', fontsize=32)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/proposed_fig_{ianomaly_type}.svg', dpi=2000)
        plt.close()

joblib.dump(knn_DL, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_KNN_DL')
joblib.dump(scaler_DL, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_scaler_DL')
torch.save(best_model, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model')
joblib.dump(knn, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_KNN')
joblib.dump(scaler, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_scaler')
joblib.dump(knn_BPFI, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_KNN_BPFI')
joblib.dump(scaler_BPFI, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_scaler_BPFI')
joblib.dump(knn_BPFO, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_KNN_BPFO')
joblib.dump(scaler_BPFO, f'./dataset_prepared/vib_1x_MLv2_1f2f3f4f_wphase_cluster_gmm_BPFIBPFO_sepv2_1dcnn_KNN_plusBPFO/{rseed}/best_model_scaler_BPFO')
