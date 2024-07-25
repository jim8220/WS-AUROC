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
import math
import torchaudio
import sys

import torch.nn.functional as F

from librosa.core.audio import __audioread_load

from matplotlib import pyplot as plt


def plot_confusion_matrix(y_true, y_pred, path, title='Confusion Matrix'):
    """
    주어진 실제 레이블과 예측 레이블을 바탕으로 혼동 행렬을 생성하고 시각화하는 함수입니다.

    Parameters:
    y_true (array): 실제 레이블
    y_pred (array): 예측 레이블
    title (str): 그래프 제목

    Returns:
    None
    """
    # 혼동 행렬 생성
    cm = metrics.confusion_matrix(y_true, y_pred)

    # 혼동 행렬 출력
    print("Confusion Matrix:")
    print(cm)

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(y_true)))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)

    # 라벨 추가
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 각 셀에 값 추가
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(path)


def plot_confusion_matrix2(y_true, y_pred, path, title='Confusion Matrix', tick_marksx=[]):
    """
    주어진 실제 레이블과 예측 레이블을 바탕으로 혼동 행렬을 생성하고 시각화하는 함수입니다.

    Parameters:
    y_true (array): 실제 레이블
    y_pred (array): 예측 레이블
    title (str): 그래프 제목

    Returns:
    None
    """
    # 혼동 행렬 생성
    cm = metrics.confusion_matrix(y_true, y_pred)

    # 혼동 행렬 출력
    print("Confusion Matrix:")
    print(cm)

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(tick_marksx))
    plt.xticks(tick_marks, tick_marksx)
    plt.yticks(tick_marks, tick_marksx)

    # 라벨 추가
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 각 셀에 값 추가
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(path)

def plot_confusion_matrix_ad(y_true, y_pred, path, title='Confusion Matrix'):
    """
    주어진 실제 레이블과 예측 레이블을 바탕으로 혼동 행렬을 생성하고 시각화하는 함수입니다.

    Parameters:
    y_true (array): 실제 레이블
    y_pred (array): 예측 레이블
    title (str): 그래프 제목

    Returns:
    None
    """
    # 혼동 행렬 생성
    cm = metrics.confusion_matrix(y_true, y_pred)

    # 혼동 행렬 출력
    print("Confusion Matrix:")
    print(cm)

    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(set(y_true)))
    plt.xticks(tick_marks, ['normal', 'anomaly'])
    plt.yticks(tick_marks, ['normal', 'anomaly'])

    # 라벨 추가
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 각 셀에 값 추가
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(path)



# ==============================================================================
# refer to DCASE2022 Task2 Top-1
# https://dcase.community/documents/challenge2022/technical_reports/DCASE2022_Liu_8_t2.pdf
Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc = nn.Linear(256 * 62, 128)  # Adjust the input size based on the final output size of the conv layers

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

class vib_MFN_feat_shared_1000hz(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_MFN_feat_shared_1000hz, self).__init__()
        #self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.cnn1d = CNN1D()
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        #self.fc_clf_2 = nn.Linear(128, num_classes_2)
        #self.fc_clf_3 = nn.Linear(128, num_classes_3)

    def forward(self, x_npy): #x_npy has dimension of B*T
        #x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        #feat1 = self.mobilefacenet(x_mel)
        x_fft = torch.abs(torch.fft.fft(x_npy.float())[:,:1000].unsqueeze(1))
        feat = self.cnn1d(x_fft)
        #feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        #out2 = self.fc_clf_2(feat)
        #out3 = self.fc_clf_3(feat)
        return out

    def get_feature(self, x_npy): #x_npy has dimension of B*T
        #x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        #feat1 = self.mobilefacenet(x_mel)
        x_fft = torch.abs(torch.fft.fft(x_npy.float())[:,:1000].unsqueeze(1))
        feat = self.cnn1d(x_fft)
        #feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        #out2 = self.fc_clf_2(feat)
        #out3 = self.fc_clf_3(feat)
        return feat

class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
        #self.dropout = nn.Dropout2d(p=0.25) #0.2~0.3
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            #x = self.dropout(x)
            return x
        else:
            x = self.prelu(x)
            #x = self.dropout(x)
            return x

class Bottleneck_MFN(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck_MFN, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class MobileFaceNet_CEE(nn.Module):
    def __init__(self,
                 input_ch,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet_CEE, self).__init__()

        self.conv1 = ConvBlock(input_ch, 64, 3, 2, 1) # ch 2 -> 1

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck_MFN
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (8, 4), 1, 0, dw=True, linear=True) # (33, 13) for STFT
        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)


        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        out = x.view(x.shape[0],-1)
        return out



class MobileFaceNet_CEE_RP(nn.Module):
    def __init__(self,
                 input_ch,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(MobileFaceNet_CEE_RP, self).__init__()

        self.conv1 = ConvBlock(input_ch, 64, 3, 2, 1) # ch 2 -> 1

        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck_MFN
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)
        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (38, 38), 1, 0, dw=True, linear=True) # (33, 13) for STFT
        # self.linear7 = ConvBlock(512, 512, (4, 10), 1, 0, dw=True, linear=True)
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)


        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        out = x.view(x.shape[0],-1)
        return out

class vib_MFN_feat_shared(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_MFN_feat_shared, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy): #x_npy has dimension of B*T
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return feat

class vib_MFN_feat_shared_1000hz_mm(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_MFN_feat_shared_1000hz_mm, self).__init__()
        self.cnn1d = CNN1D()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)


    def forward(self, x_npy): #x_npy has dimension of B*T
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        x_fft = torch.abs(torch.fft.fft(x_npy.float())[:, :1000].unsqueeze(1))
        feat = self.mobilefacenet(x_mel)
        feat_fft = self.cnn1d(x_fft)
        out = self.fc_clf(torch.cat([feat, feat_fft], dim=1))
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        x_fft = torch.abs(torch.fft.fft(x_npy.float())[:, :1000].unsqueeze(1))
        feat = self.mobilefacenet(x_mel)
        feat_fft = self.cnn1d(x_fft)
        out = self.fc_clf(torch.cat([feat, feat_fft], dim=1))
        return out

class vib_MFN_feat_shared_MTL(nn.Module):
    def __init__(self, num_classes, num_classes_2,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_MFN_feat_shared_MTL, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.fc_clf_2 = nn.Linear(128, num_classes_2)

    def forward(self, x_npy): #x_npy has dimension of B*T
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        out2 = self.fc_clf_2(feat)
        return out, out2

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        out2 = self.fc_clf_2(feat)
        return feat

class vib_MFN_feat_shared_MTL_manyclass(nn.Module):
    def __init__(self, num_classes, num_classes_2, num_classes_3,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_MFN_feat_shared_MTL_manyclass, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.fc_clf_2 = nn.Linear(128, num_classes_2)
        self.fc_clf_3 = nn.Linear(128, num_classes_3)

    def forward(self, x_npy): #x_npy has dimension of B*T
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        out2 = self.fc_clf_2(feat)
        out3 = self.fc_clf_3(feat)
        return out, out2, out3

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy.float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        out2 = self.fc_clf_2(feat)
        out3 = self.fc_clf_3(feat)
        return feat


class cur_MFN_feat_shared_MTL_manyclass(nn.Module):
    def __init__(self, num_classes, num_classes_2, num_classes_3,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(cur_MFN_feat_shared_MTL_manyclass, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE_RP(input_ch=1, bottleneck_setting=bottleneck_setting)
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.fc_clf_2 = nn.Linear(128, num_classes_2)
        self.fc_clf_3 = nn.Linear(128, num_classes_3)
        self.N_thres = 6

    def forward(self, x_npy): #x_npy has dimension of B*T
        x_cur = x_npy[:,:600]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        out2 = self.fc_clf_2(feat)
        out3 = self.fc_clf_3(feat)
        return out, out2, out3

    def get_feature(self, x_npy):
        x_cur = x_npy[:,:600]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        out2 = self.fc_clf_2(feat)
        out3 = self.fc_clf_3(feat)
        return feat

class vibxA_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibxA_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,5].float()) + sys.float_info.epsilon).unsqueeze(1)
        out = self.mobilefacenet(x_mel)
        out = self.fc_clf(out)
        return out

class vibxA_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibxA_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,5].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,5].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return feat

class TgramNet2(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet2, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(201),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)]
        )
        # GAP Layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(mel_bins, 128)

    def forward(self, x):
        out = self.conv_extractor(x)
        out = self.conv_encoder(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out


class TemporalAttention(nn.Module):
    def __init__(self, in_dim):
        super(TemporalAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, T = x.size()
        proj_query = self.query_conv(x).view(batch_size, -1, T).permute(0, 2, 1)  # B x T x C
        proj_key = self.key_conv(x).view(batch_size, -1, T)  # B x C x T
        energy = torch.bmm(proj_query, proj_key)  # B x T x T
        attention = F.softmax(energy, dim=-1)  # B x T x T
        proj_value = self.value_conv(x).view(batch_size, -1, T)  # B x C x T

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x T
        out = out.view(batch_size, C, T)

        out = self.gamma * out + x
        return out


class TgramNet3(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet3, self).__init__()
        self.conv_extractor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(201),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)]
        )
        self.temporal_attention = TemporalAttention(mel_bins)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(mel_bins, 128)

    def forward(self, x):
        out = self.conv_extractor(x)
        out = self.conv_encoder(out)
        out = self.temporal_attention(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        return out

class vibfftxA_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibfftxA_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram2 = TgramNet2()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,5].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,5].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,5].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,5].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class vibattfftxA_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibattfftxA_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram3 = TgramNet3()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,5].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,5].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,5].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,5].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class vibyA_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibyA_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,6].float()) + sys.float_info.epsilon).unsqueeze(1)
        out = self.mobilefacenet(x_mel)
        out = self.fc_clf(out)
        return out

class vibfftyA_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibfftyA_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram2 = TgramNet2()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,6].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,6].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,6].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,6].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class vibyA_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibyA_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,6].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,6].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return feat

class vibattfftyA_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibattfftyA_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram3 = TgramNet3()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,6].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,6].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,6].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,6].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class vibxB_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibxB_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,7].float()) + sys.float_info.epsilon).unsqueeze(1)
        out = self.mobilefacenet(x_mel)
        out = self.fc_clf(out)
        return out

class vibxB_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibxB_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,7].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,7].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return feat

class vibfftxB_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibfftxB_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram2 = TgramNet2()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,7].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,7].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,7].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,7].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class vibattfftxB_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibattfftxB_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram3 = TgramNet3()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,7].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,7].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,7].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,7].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class vibyB_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibyB_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out = self.mobilefacenet(x_mel)
        out = self.fc_clf(out)
        return out

class vibyB_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibyB_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(x_mel)
        out = self.fc_clf(feat)
        return feat

class vibfftyB_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibfftyB_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram2 = TgramNet2()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,8].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram2(x_npy[:,:,8].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class vibattfftyB_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibattfftyB_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)
        self.Tgram3 = TgramNet3()

    def forward(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,8].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel = 10 * torch.log10(self.mel(x_npy[:,:,8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat1 = self.mobilefacenet(x_mel)
        feat2 = self.Tgram3(x_npy[:,:,8].float().unsqueeze(1))
        feat = torch.cat([feat1, feat2], dim=1)
        out = self.fc_clf(feat)
        return feat

class curU_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(curU_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE_RP(input_ch=1, bottleneck_setting=bottleneck_setting)
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.N_thres = 6


    def forward(self, x_npy):
        x_cur = x_npy[:,:600,2]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        out = self.mobilefacenet(x_RP)
        out = self.fc_clf(out)
        return out

class curU_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(curU_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE_RP(input_ch=1, bottleneck_setting=bottleneck_setting)
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.N_thres = 6


    def forward(self, x_npy):
        x_cur = x_npy[:,:600,2]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_cur = x_npy[:,:600,2]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        return feat

class curV_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(curV_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE_RP(input_ch=1, bottleneck_setting=bottleneck_setting)
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.N_thres = 6


    def forward(self, x_npy):
        x_cur = x_npy[:,:600,3]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        out = self.mobilefacenet(x_RP)
        out = self.fc_clf(out)
        return out


class curV_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(curV_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE_RP(input_ch=1, bottleneck_setting=bottleneck_setting)
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.N_thres = 6


    def forward(self, x_npy):
        x_cur = x_npy[:,:600,3]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_cur = x_npy[:,:600,3]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        return feat

class curW_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(curW_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE_RP(input_ch=1, bottleneck_setting=bottleneck_setting)
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.N_thres = 6


    def forward(self, x_npy):
        x_cur = x_npy[:,:600,4]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        out = self.mobilefacenet(x_RP)
        out = self.fc_clf(out)
        return out


class curW_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(curW_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE_RP(input_ch=1, bottleneck_setting=bottleneck_setting)
        #self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)
        self.N_thres = 6


    def forward(self, x_npy):
        x_cur = x_npy[:,:600,4]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_cur = x_npy[:,:600,4]
        x_RP= torch.min(torch.abs(x_cur.unsqueeze(1) - x_cur.unsqueeze(2)),torch.tensor(self.N_thres))
        x_RP = x_RP.unsqueeze(1).float()
        feat = self.mobilefacenet(x_RP)
        out = self.fc_clf(feat)
        return feat

class viball_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(viball_MFN, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=4, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out = self.mobilefacenet(torch.cat([x_mel1, x_mel2, x_mel3, x_mel4], dim=1))
        out = self.fc_clf(out)
        return out

class viball_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(viball_MFN_feat, self).__init__()
        self.mobilefacenet = MobileFaceNet_CEE(input_ch=4, bottleneck_setting=bottleneck_setting)
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128, num_classes)


    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(torch.cat([x_mel1, x_mel2, x_mel3, x_mel4], dim=1))
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        feat = self.mobilefacenet(torch.cat([x_mel1, x_mel2, x_mel3, x_mel4], dim=1))
        out = self.fc_clf(feat)
        return feat

class vibB_fusion_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vibB_fusion_MFN, self).__init__()
        self.mobilefacenet1 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet2 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)

        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*2, num_classes)


    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out1 = self.mobilefacenet1(x_mel1)
        out2 = self.mobilefacenet2(x_mel2)
        out = torch.cat([out1, out2], dim=1)
        out = self.fc_clf(out)
        return out

class vib_fusion_MFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_fusion_MFN, self).__init__()
        self.mobilefacenet1 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet2 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet3 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet4 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)

        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*4, num_classes)


    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out1 = self.mobilefacenet1(x_mel1)
        out2 = self.mobilefacenet2(x_mel2)
        out3 = self.mobilefacenet3(x_mel3)
        out4 = self.mobilefacenet4(x_mel4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.fc_clf(out)
        return out

class vib_fusion_MFN_feat(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_fusion_MFN_feat, self).__init__()
        self.mobilefacenet1 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet2 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet3 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet4 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)

        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*4, num_classes)


    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out1 = self.mobilefacenet1(x_mel1)
        out2 = self.mobilefacenet2(x_mel2)
        out3 = self.mobilefacenet3(x_mel3)
        out4 = self.mobilefacenet4(x_mel4)
        feat = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.fc_clf(feat)
        return out

    def get_feature(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out1 = self.mobilefacenet1(x_mel1)
        out2 = self.mobilefacenet2(x_mel2)
        out3 = self.mobilefacenet3(x_mel3)
        out4 = self.mobilefacenet4(x_mel4)
        feat = torch.cat([out1, out2, out3, out4], dim=1)
        return feat

class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                nn.LayerNorm(201),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out

class vib_fusion_STgramMFN(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_fusion_STgramMFN, self).__init__()
        self.mobilefacenet1 = MobileFaceNet_CEE(input_ch=2, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet2 = MobileFaceNet_CEE(input_ch=2, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet3 = MobileFaceNet_CEE(input_ch=2, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet4 = MobileFaceNet_CEE(input_ch=2, bottleneck_setting=bottleneck_setting)

        self.tgramnet1 = TgramNet()
        self.tgramnet2 = TgramNet()
        self.tgramnet3 = TgramNet()
        self.tgramnet4 = TgramNet()


        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*4, num_classes)


    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        xt1 = self.tgramnet1(x_npy[:, :, 5].unsqueeze(1).float())
        xt2 = self.tgramnet2(x_npy[:, :, 6].unsqueeze(1).float())
        xt3 = self.tgramnet3(x_npy[:, :, 7].unsqueeze(1).float())
        xt4 = self.tgramnet4(x_npy[:, :, 8].unsqueeze(1).float())
        out1 = self.mobilefacenet1(torch.cat([x_mel1, xt1.unsqueeze(1)], dim=1))
        out2 = self.mobilefacenet2(torch.cat([x_mel2, xt2.unsqueeze(1)], dim=1))
        out3 = self.mobilefacenet3(torch.cat([x_mel3, xt3.unsqueeze(1)], dim=1))
        out4 = self.mobilefacenet4(torch.cat([x_mel4, xt4.unsqueeze(1)], dim=1))
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.fc_clf(out)
        return out

class vib_fusion_STgramMFN_v2(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_fusion_STgramMFN_v2, self).__init__()
        self.mobilefacenet1 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet2 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet3 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet4 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)

        self.mobilefacenet1t = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet2t = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet3t = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet4t = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)


        self.tgramnet1 = TgramNet()
        self.tgramnet2 = TgramNet()
        self.tgramnet3 = TgramNet()
        self.tgramnet4 = TgramNet()


        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*8, num_classes)


    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        xt1 = self.tgramnet1(x_npy[:, :, 5].unsqueeze(1).float())
        xt2 = self.tgramnet2(x_npy[:, :, 6].unsqueeze(1).float())
        xt3 = self.tgramnet3(x_npy[:, :, 7].unsqueeze(1).float())
        xt4 = self.tgramnet4(x_npy[:, :, 8].unsqueeze(1).float())
        out1 = self.mobilefacenet1(x_mel1)
        out2 = self.mobilefacenet2(x_mel2)
        out3 = self.mobilefacenet3(x_mel3)
        out4 = self.mobilefacenet4(x_mel4)
        outt1 = self.mobilefacenet1t(xt1.unsqueeze(1))
        outt2 = self.mobilefacenet2t(xt2.unsqueeze(1))
        outt3 = self.mobilefacenet3t(xt3.unsqueeze(1))
        outt4 = self.mobilefacenet4t(xt4.unsqueeze(1))
        out = torch.cat([out1, out2, out3, out4, outt1, outt2, outt3, outt4], dim=1)
        out = self.fc_clf(out)
        return out

class vib_fusion_MFN_each(nn.Module):
    def __init__(self, num_classes,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting):
        super(vib_fusion_MFN_each, self).__init__()
        self.mobilefacenet1 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet2 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet3 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)
        self.mobilefacenet4 = MobileFaceNet_CEE(input_ch=1, bottleneck_setting=bottleneck_setting)

        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=25600,n_fft=1024,hop_length=512,n_mels=128,power=2)
        #self.stft = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=512, power=2)
        self.fc_clf = nn.Linear(128*4, num_classes)
        self.fc_clf_each1 = nn.Linear(128, num_classes)
        self.fc_clf_each2 = nn.Linear(128, num_classes)
        self.fc_clf_each3 = nn.Linear(128, num_classes)
        self.fc_clf_each4 = nn.Linear(128, num_classes)



    def forward(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out1 = self.mobilefacenet1(x_mel1)
        out2 = self.mobilefacenet2(x_mel2)
        out3 = self.mobilefacenet3(x_mel3)
        out4 = self.mobilefacenet4(x_mel4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.fc_clf(out)
        out1 = self.fc_clf_each1(out1)
        out2 = self.fc_clf_each2(out2)
        out3 = self.fc_clf_each3(out3)
        out4 = self.fc_clf_each4(out4)
        return out, out1, out2, out3, out4

    def inference(self, x_npy):
        x_mel1 = 10 * torch.log10(self.mel(x_npy[:, :, 5].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel2 = 10 * torch.log10(self.mel(x_npy[:, :, 6].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel3 = 10 * torch.log10(self.mel(x_npy[:, :, 7].float()) + sys.float_info.epsilon).unsqueeze(1)
        x_mel4 = 10 * torch.log10(self.mel(x_npy[:, :, 8].float()) + sys.float_info.epsilon).unsqueeze(1)
        out1 = self.mobilefacenet1(x_mel1)
        out2 = self.mobilefacenet2(x_mel2)
        out3 = self.mobilefacenet3(x_mel3)
        out4 = self.mobilefacenet4(x_mel4)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        out = self.fc_clf(out)
        return out