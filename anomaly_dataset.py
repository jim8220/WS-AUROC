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
import glob
import random
from librosa.core.audio import __audioread_load
from tqdm import tqdm


def anomaly_dataset(rseed=0):
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
    # this returns numpy array of file list
    # input

    # rseed is randomseed

    # output
    # should return train data, valid data, test data

    # load file list from ./processed_data/{machine_type}
    # here, same index with microphone should be paired


    flist_normal = glob.glob(f'./processed_data/normal/*')
    flist_normal.sort(key=os.path.getmtime)
    np.random.shuffle(flist_normal)
    normal_paired_flist = flist_normal

    # hyper parameter for dividing train / valid / test
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    # build train / valid / test file list

    train_file_list = []
    valid_file_list = []
    test_file_list = []

    train_file_list += normal_paired_flist[:int(len(normal_paired_flist)*train_ratio)]
    valid_file_list += normal_paired_flist[int(len(normal_paired_flist)*train_ratio):int(len(normal_paired_flist)*(train_ratio+valid_ratio))]
    test_file_list += normal_paired_flist[int(len(normal_paired_flist)*(train_ratio+valid_ratio)):]

    # for test, anomaly should contain same amout of anomaly data for each anomaly types

    test_normal_len = len(test_file_list)

    flist_anomaly_UB = glob.glob(f'./processed_data/anomaly/*Unbalance*')
    test_file_list += flist_anomaly_UB[:test_normal_len]
    flist_anomaly_BPFI = glob.glob(f'./processed_data/anomaly/*BPFI*')
    test_file_list += flist_anomaly_BPFI[:test_normal_len]

    flist_anomaly_MA = glob.glob(f'./processed_data/anomaly/*Misalign*')
    test_file_list += flist_anomaly_MA[:test_normal_len]
    # BPFO data can not be used (정원호 박사과정님과 데이터를 좀 더 수집해야 할 수 도 있음)

    train_npy_list = []
    valid_npy_list = []
    test_npy_list = []

    for ifidx in tqdm(range(len(train_file_list)), desc = "train_npy_loading"):

        train_npy_list.append(np.load(train_file_list[ifidx]))

    for ifidx in tqdm(range(len(valid_file_list)), desc = "valid_npy_loading"):

        valid_npy_list.append(np.load(valid_file_list[ifidx]))

    for ifidx in tqdm(range(len(test_file_list)), desc = "test_npy_loading"):

        test_npy_list.append(np.load(test_file_list[ifidx]))

    # convert to numpy array

    train_npy_list_np = np.asarray(train_npy_list)
    valid_npy_list_np = np.asarray(valid_npy_list)
    test_npy_list_np = np.asarray(test_npy_list)

    return train_npy_list_np, valid_npy_list_np, test_npy_list_np, train_file_list, valid_file_list, test_file_list



def anomaly_dataset_extended(rseed=0):
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
    # this returns numpy array of file list
    # input

    # rseed is randomseed

    # output
    # should return train data, valid data, test data

    # load file list from ./processed_data/{machine_type}
    # here, same index with microphone should be paired


    flist_normal = glob.glob(f'./processed_data/normal/*')
    flist_normal.sort(key=os.path.getmtime)
    np.random.shuffle(flist_normal)
    normal_paired_flist = flist_normal

    # hyper parameter for dividing train / valid / test
    train_ratio = 0.8
    valid_ratio = 0.1
    test_ratio = 0.1

    # build train / valid / test file list

    train_file_list = []
    valid_file_list = []
    test_file_list = []

    train_file_list += normal_paired_flist[:int(len(normal_paired_flist)*train_ratio)]
    valid_file_list += normal_paired_flist[int(len(normal_paired_flist)*train_ratio):int(len(normal_paired_flist)*(train_ratio+valid_ratio))]
    test_file_list += normal_paired_flist[int(len(normal_paired_flist)*(train_ratio+valid_ratio)):]

    # for test, anomaly should contain same amout of anomaly data for each anomaly types

    test_normal_len = len(test_file_list)

    flist_anomaly_UB = glob.glob(f'./processed_data/anomaly/*Unbalance*')
    test_file_list += flist_anomaly_UB[:test_normal_len]
    flist_anomaly_BPFI = glob.glob(f'./processed_data/anomaly/*BPFI*')
    test_file_list += flist_anomaly_BPFI[:test_normal_len]
    flist_anomaly_BPFO = glob.glob(f'./processed_data/anomaly/*BPFO*')
    test_file_list += flist_anomaly_BPFO[:test_normal_len]

    flist_anomaly_MA = glob.glob(f'./processed_data/anomaly/*Misalign*')
    test_file_list += flist_anomaly_MA[:test_normal_len]
    # BPFO data can not be used (정원호 박사과정님과 데이터를 좀 더 수집해야 할 수 도 있음)

    train_npy_list = []
    valid_npy_list = []
    test_npy_list = []

    for ifidx in tqdm(range(len(train_file_list)), desc = "train_npy_loading"):

        train_npy_list.append(np.load(train_file_list[ifidx]))

    for ifidx in tqdm(range(len(valid_file_list)), desc = "valid_npy_loading"):

        valid_npy_list.append(np.load(valid_file_list[ifidx]))

    for ifidx in tqdm(range(len(test_file_list)), desc = "test_npy_loading"):

        test_npy_list.append(np.load(test_file_list[ifidx]))

    # convert to numpy array

    train_npy_list_np = np.asarray(train_npy_list)
    valid_npy_list_np = np.asarray(valid_npy_list)
    test_npy_list_np = np.asarray(test_npy_list)

    return train_npy_list_np, valid_npy_list_np, test_npy_list_np, train_file_list, valid_file_list, test_file_list
