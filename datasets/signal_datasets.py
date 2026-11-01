import os
import json
import cv2
import numpy as np
import scipy.io as sio
import torch
import h5py
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, data_dir, train=True, rppg_len=160):
        self.data_dir = data_dir
        self.train = train
        self.rppg_len = rppg_len
        self.data_list = list()
        self.get_data_list()

    def get_data_list(self, rppg_len):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):        
        rppg = self.data_list[index]
        return rppg

    def calculate_chunk_split(self, old_length, old_rate, new_rate, T):
        if T == -1:
            return [[0, old_length]]
        else:
            new_length = round(old_length * new_rate / old_rate) 
            if new_length < T:
                raise ValueError("video length is too short")
            else:
                map_to_old = int(T * old_rate / new_rate) + 1
                chunk_split = list(range(0, old_length - map_to_old + 1, map_to_old)) #
                chunk_list = []
                for chunk in chunk_split:
                    if chunk + map_to_old <= old_length:
                        chunk_list.append([chunk, chunk + map_to_old])
            return chunk_list


class PURE(BaseDataset):
    def __init__(self, data_dir, train=True, rppg_len=160):
        super().__init__(data_dir, train, rppg_len)

    def get_data_list(self):
        date_list = os.listdir(self.data_dir)
        date_list.sort()
        # Subject-disjoint 7/3 protocol: subjects 01--07 for training and
        # subjects 08--10 for testing.  (06-02 is absent from PURE.)
        train_list = [date for date in date_list if int(date.split('-')[0]) <= 7]
        if self.train:
            date_list = [i for i in date_list if i in train_list]
        else:
            date_list = [i for i in date_list if i not in train_list]
        for date in date_list:
            # read label
            json_file = os.path.join(self.data_dir, date, date + ".json")
            with open(json_file, 'r') as f:
                data = json.load(f)
            ecg_time_stamp = np.array([i['Timestamp'] for i in data['/FullPackage']])
            ecg = np.array([i['Value']['waveform'] for i in data['/FullPackage']])
            assert len(ecg_time_stamp) == len(ecg)
            ecg_time_diffs = np.diff(ecg_time_stamp / 1e9)
            ecg_rate = 1 / ecg_time_diffs.mean()
            ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, 30, self.rppg_len)
            for idx in range(len(ecg_chunk_split)):
                start_idx, end_idx = ecg_chunk_split[idx]
                rppg_sample = ecg[start_idx:end_idx]
                rppg_sample = rppg_sample - np.mean(rppg_sample)
                rppg_sample = rppg_sample / np.max(np.abs(rppg_sample))
                rppg_sample = np.interp(np.linspace(0, self.rppg_len, self.rppg_len), np.linspace(0, self.rppg_len, len(rppg_sample)), rppg_sample)
                rppg_sample = torch.from_numpy(rppg_sample).float()
                self.data_list.append(rppg_sample)


class UBFC(BaseDataset):
    def __init__(self, data_dir='', train=True, rppg_len=160):
        super().__init__(data_dir, train=train, rppg_len=rppg_len)

    def get_data_list(self):
        subject_list = os.listdir(self.data_dir)
        subject_list.remove('subject11')
        subject_list.remove('subject18')
        subject_list.remove('subject20')
        subject_list.remove('subject24') 
        subject_list.sort()
        if self.train:
            subject_list = subject_list[:30]
        else:
            subject_list = subject_list[30:]
        for subject in subject_list:
            file_dir = os.path.join(self.data_dir, subject)
            # read label
            with open(os.path.join(file_dir, 'ground_truth.txt'), 'r') as f:
                data = f.readlines()
            data_timestamp = np.array([float(strr.replace('e','E')) for strr in list(data[2].split())])
            data_Hr = np.array([float(strr.replace('e','E')) for strr in list(data[1].split())])
            data_ecg = np.array([float(strr.replace('e','E')) for strr in list(data[0].split())])
            assert len(data_timestamp) == len(data_Hr) == len(data_ecg)
            time_diffs = np.diff(data_timestamp)
            frame_rate = 1 / time_diffs.mean()
            chunk_split = self.calculate_chunk_split(len(data_ecg), frame_rate, 30, self.rppg_len)
            for start_idx, end_idx in chunk_split:
                rppg_sample = data_ecg[start_idx:end_idx]
                rppg_sample = rppg_sample - np.mean(rppg_sample)
                rppg_sample = rppg_sample / np.max(np.abs(rppg_sample))
                rppg_sample = np.interp(np.linspace(0, self.rppg_len, self.rppg_len), np.linspace(0, self.rppg_len, len(rppg_sample)), rppg_sample)
                rppg_sample = torch.from_numpy(rppg_sample).float()
                self.data_list.append(rppg_sample)


class VIPL(BaseDataset):
    def __init__(self, data_dir = '', train=True, rppg_len=160, fold=5):
        self.fold = fold
        super().__init__(data_dir, train=train, rppg_len=rppg_len)

    def get_data_list(self):
        self.fold_split_dir = os.path.join(self.data_dir, "VIPL_fold")
        self.fold_list = []
        for i in range(1, 6):
            mat_path = os.path.join(self.fold_split_dir, f"fold{i}.mat")
            mat = sio.loadmat(mat_path)
            self.fold_list.append(mat[f"fold{i}"].reshape(-1))

        if self.train:
            # all flod except self.fold
            fold = np.concatenate(self.fold_list[:self.fold - 1] + self.fold_list[self.fold:])
        else:
            fold = self.fold_list[self.fold - 1]

        # print(fold)
        p_lists = [f"p{i}" for i in fold]
        p_lists.sort()
        for p_name in p_lists:
            p_root = os.path.join(self.data_dir, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                source_lists = os.listdir(v_root)
                if "source4" in source_lists:
                    source_lists.remove("source4")
                source_lists.sort()
                for source_name in source_lists:                    
                    if os.path.join(v_root, source_name) in [f'{self.data_dir}/p32/v7/source3', f'{self.data_dir}/p45/v1/source2', \
                                    f'{self.data_dir}/p19/v2/source2']: # 32-7-3, 45-1-2 lack of wave, 19-2-2 lack of frame 
                        continue
                    # read label
                    wave_csv = os.path.join(v_root, source_name, "wave.csv")
                    with open(wave_csv, 'r') as f:
                        data = f.readlines()
                        data = data[1:]
                        ecg = np.array([int(i) for i in data])
                    # read time.txt calculate frame_rate, ecg_rate
                    rate_txt = os.path.join(v_root, source_name, "rate.txt")
                    with open(rate_txt, 'r') as f:
                        data = f.read().splitlines()
                        # print(data)
                        total_time = float(data[2].split(":")[1])
                    ecg_rate = len(ecg) / total_time
                    try: # in case of frame_number does not match with wave
                        ecg_chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, 30, self.rppg_len)
                    except:
                        continue
                    # add rppg label
                    for idx in range(len(ecg_chunk_split)):
                        ### sample rppg
                        start_idx, end_idx = ecg_chunk_split[idx]
                        rppg_sample = ecg[start_idx:end_idx]
                        rppg_sample = rppg_sample - np.mean(rppg_sample)
                        rppg_sample = rppg_sample / np.max(np.abs(rppg_sample))
                        rppg_sample = np.interp(np.linspace(0, self.rppg_len, self.rppg_len), np.linspace(0, self.rppg_len, len(rppg_sample)), rppg_sample)
                        rppg_sample = torch.from_numpy(rppg_sample).float()
                        self.data_list.append(rppg_sample)


class COHFACE(BaseDataset):
    def __init__(self, data_dir='', train=True, rppg_len=160):
        super().__init__(data_dir, train=train, rppg_len=rppg_len)

    def get_data_list(self):
        test_txt = f'{self.data_dir}/protocols/all/test.txt'
        train_txt = f'{self.data_dir}/protocols/all/train.txt'
        
        if self.train:
            with open(train_txt, 'r') as f:
                all_list = f.readlines()
        else:
            with open(test_txt, 'r') as f:
                all_list = f.readlines()

        for data_sample in all_list:
            [px, v_src] = data_sample.strip().split('/')[:2]
            px_path = os.path.join(self.data_dir, px)
            if (px, v_src) in [('11', '2'), ('3', '2'), ('3', '3'), ('6', '2')]:
                continue
            # read label
            with h5py.File(os.path.join(px_path, v_src, 'data.hdf5'), 'r') as f:
                ecg = np.array(f['pulse'], dtype=np.float32)
                time = np.array(f['time'], dtype=np.float32)
            ecg = ecg[20:]
            ecg_rate = ecg.shape[0] / time[-1]
            chunk_split = self.calculate_chunk_split(len(ecg), ecg_rate, 30, self.rppg_len)
            for start_idx, end_idx in chunk_split:
                rppg_sample = ecg[start_idx:end_idx]
                rppg_sample = rppg_sample - np.mean(rppg_sample)
                rppg_sample = rppg_sample / np.max(np.abs(rppg_sample))
                rppg_sample = np.interp(np.linspace(0, self.rppg_len, self.rppg_len), np.linspace(0, self.rppg_len, len(rppg_sample)), rppg_sample)
                rppg_sample = torch.from_numpy(rppg_sample).float()
                self.data_list.append(rppg_sample)

