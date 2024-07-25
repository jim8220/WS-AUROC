import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io
from nptdms import TdmsFile

os.makedirs('./processed_data', exist_ok=True)
os.makedirs('./processed_data/normal', exist_ok=True)
os.makedirs('./processed_data/anomaly', exist_ok=True)
normal_file_lists = [iname[:-4] for iname in os.listdir('./vibration') if 'Normal' in iname]
anomaly_file_lists = [iname[:-4] for iname in os.listdir('./vibration') if 'Normal' not in iname]

num_lists = {}

error_file_names = []

for file_name in normal_file_lists:

    # convert tdms file to numpy array

    # convert mat file to numpy array

    mat_data = scipy.io.loadmat(f'./vibration/{file_name}.mat')
    vib = mat_data['Signal'][0][0][1][0][0][0]

    # concatenate all data in single numpy array

    # 이 부분은 정원호 박사과정님과 논의하여 수정
    # dataset이 비동기화 되어서 생기는 문제로 파악
    # 우선은 끝에 부분을 잘라내서 맞추는 쪽으로


    temp_cur_vib = vib

    fs = 25600  # sampling freq 25.6 kHz
    tlen = 1  # time length
    tt = tlen * fs
    # cut by 1 s

    nsample = len(temp_cur_vib) // tt
    # cut by 1 s

    for isample in tqdm(range(nsample), desc=f'saving {file_name}'):
        np.save(f'./processed_data/normal/{file_name}_{isample+1}.npy', temp_cur_vib[isample*tt:(isample+1)*tt,:])

    num_lists[file_name] = nsample

for file_name in anomaly_file_lists:

    # convert tdms file to numpy array
    try:

        # typo (파일 이름 Unbalance -> Unbalalnce)
        if '2Nm_Unbalance' in file_name:
            file_name = file_name.replace('Unbalance', 'Unbalalnce')

        mat_data = scipy.io.loadmat(f'./vibration/{file_name}.mat')
        vib = mat_data['Signal'][0][0][1][0][0][0]

        if 'Unbalalnce' in file_name:
            file_name = file_name.replace('Unbalalnce', 'Unbalance')
        # concatenate all data in single numpy array

        # 이 부분은 정원호 박사과정님과 논의하여 수정
        # dataset이 비동기화 되어서 생기는 문제로 파악
        # 우선은 끝에 부분을 잘라내서 맞추는 쪽으로

        temp_cur_vib = vib

        fs = 25600  # sampling freq 25.6 kHz
        tlen = 1 # time length
        tt = tlen*fs
        # cut by 1 s

        nsample = len(temp_cur_vib) // tt

        for isample in tqdm(range(nsample), desc=f'saving {file_name}'):
            np.save(f'./processed_data/anomaly/{file_name}_{isample + 1}.npy', temp_cur_vib[isample * tt:(isample + 1) * tt, :])

        num_lists[file_name] = nsample
    except:
        error_file_names.append(file_name)

# save error file names (mismatch of numbers of lists-> usually empty case)

with open('./processed_data/error_file_names.txt', 'w') as file:
    file.write(','.join(map(str, error_file_names)))

# save bar plot

labels = list(num_lists.keys())
values = list(num_lists.values())

plt.figure(figsize=(12, 8))
plt.bar(labels, values)
plt.xticks(rotation=90)
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph of num_lists Data')

plt.tight_layout()
plt.savefig('./processed_data/num_lists_bar_graph.png')
