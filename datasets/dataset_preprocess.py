'''
preprocess all rppg datasets: 
    read all rppg signals & videos and save them as hdf5 files (with fps -> 30)
for each .hdf5 file:
    "video_data" : (C, T, H, W), T = 30 / origin_fps * origin_T
    "ecg_data" : (T), T = 30 / origin_ecg_fps * origin_ecg_T
'''

import os
import cv2
import torch
import argparse
import h5py
import json
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

class PreprocessRppgDatasets:
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.dataset_dir = args.dataset_dir
        self.h = args.h
        self.w = args.w

    def read_video(self, frame_path, cur_fps, target_fps=30):
        # read video
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        video_x = np.zeros((len(frame_path), self.h, self.w, 3))
        for i, frame in enumerate(frame_path):
            imageBGR = cv2.imread(frame)
            try:
                imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
            except:
                print(f'error in {frame}')
            video_x[i, :, :, :] = cv2.resize(imageRGB, (self.h, self.w), interpolation=cv2.INTER_CUBIC) # T, H, W, C
        
        # resample when the difference of fps is larger than 1
        target_len = int(len(video_x) * target_fps / cur_fps)
        video_x_torch = torch.from_numpy(video_x.transpose(3, 0, 1, 2)).float().unsqueeze(0).cuda()    # (1, c, T, h, w)
        video_x_torch = torch.nn.functional.interpolate(video_x_torch, size=(target_len, self.h, self.w), mode='trilinear', align_corners=False)
        video_x = video_x_torch.squeeze(0).cpu().numpy().transpose(1, 2, 3, 0) # (T, h, w, c)
        return video_x  # (T, h, w, c)
    
    def resample_ecg(self, ecg, target_len):
        # resample when the difference of fps is larger than 1
        if len(ecg) != target_len:
            ecg_torch = torch.from_numpy(ecg).float().unsqueeze(0).unsqueeze(0)    # (1, T)
            ecg_torch = torch.nn.functional.interpolate(ecg_torch, size=(target_len), mode='linear', align_corners=False)
            ecg = ecg_torch.view(-1).numpy() # (T)
        return ecg

    def preprocess(self):
        if self.dataset == 'VIPL':
            self.preprocess_VIPL()
        elif self.dataset == 'UBFC':
            self.preprocess_UBFC()
        elif self.dataset == 'PURE':
            self.preprocess_PURE()
        elif self.dataset == 'COHFACE':
            self.preprocess_COHFACE()
        else:
            raise NotImplementedError
        
    def preprocess_VIPL(self):
        ## VIPL is special, it has wrong time of video, we should fix it
        sample_dict = {}
        for train_data in ['VIPL_ECCV_train1', 'VIPL_ECCV_train2']:
            with open(f'{self.dataset_dir}/train_val_list/{train_data}.txt', 'r') as f:
                data = f.readlines()
            for line in data:
                sample_name = line.split()[0]
                start_frame = int(line.split()[1]) - 1
                bvp_rate = float(line.split()[2])
                gt_hr = float(line.split()[3])
                bvp_signals = np.array([float(i) for i in line.split()[5:5+160]])
                if sample_name not in sample_dict:
                    sample_dict[sample_name] = {start_frame : {'bvp_rate': bvp_rate, 'bvp_signals': bvp_signals, 'gt_hr': gt_hr}}
                elif start_frame not in sample_dict[sample_name]:
                    sample_dict[sample_name][start_frame] = {'bvp_rate': bvp_rate, 'bvp_signals': bvp_signals, 'gt_hr': gt_hr}
                else:
                    continue
        for p in sorted(os.listdir(self.dataset_dir)):
            if not p.startswith('p'):
                continue
            p_path = os.path.join(self.dataset_dir, p)
            for v in sorted(os.listdir(p_path)):
                v_path = os.path.join(p_path, v)
                for source in sorted(os.listdir(v_path)):
                    if source == 'source4':
                        continue
                    sample_name = f'{p}/{v}/{source}'
                    if sample_name not in sample_dict:
                        continue
                    save_path = os.path.join(v_path, source, 'rate.txt')
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    video_path = os.path.join(v_path, source, 'align_crop_pic')
                    sample_key = min(sample_dict[sample_name].keys())
                    frame_rate = sample_dict[sample_name][sample_key]['bvp_rate']
                    total_frame = len(os.listdir(video_path))
                    total_time = total_frame / frame_rate
                    with open(save_path, 'w') as f:
                        f.write(f'frame_rate:{frame_rate}\n')
                        f.write(f'total_frame:{total_frame}\n')
                        f.write(f'total_time:{total_time}\n')
                    print(f'processing {save_path} ... \r', end='')

        ## after fixed, we could read video and ecg
        p_lists = os.listdir(self.dataset_dir)
        p_lists.sort()
        for p_name in p_lists:
            if not p_name.startswith("p"):
                continue
            p_root = os.path.join(self.dataset_dir, p_name)
            v_lists = os.listdir(p_root)
            v_lists.sort()
            for v_name in v_lists:
                v_root = os.path.join(p_root, v_name)
                source_lists = os.listdir(v_root)
                if "source4" in source_lists:
                    source_lists.remove("source4")
                source_lists.sort()
                for source_name in source_lists:
                    # read video
                    pic_type = 'align_crop_pic'    # NOTE: pic, align_crop_pic, aligned_pic
                    frame_dir = os.path.join(v_root, source_name, pic_type)
                    if frame_dir in [f'{self.dataset_dir}/p32/v7/source3/{pic_type}', f'{self.dataset_dir}/p45/v1/source2/{pic_type}', \
                                     f'{self.dataset_dir}/p19/v2/source2/{pic_type}']: # 32-7-3, 45-1-2 lack of wave, 19-2-2 lack of frame 
                        continue
                    frame_list = os.listdir(frame_dir)
                    try:
                        frame_list_int = [int(i.split(".")[0]) for i in frame_list]
                        frame_list_int.sort() # make sure the frame order is correct
                    except:
                        print(frame_dir)
                    frame_path = []
                    for frame_name in frame_list_int:
                        frame_path.append(os.path.join(frame_dir, f"{frame_name:0>5}.png")) # NOTE: pic : frame_name, align_crop_pic : frame_name:0>5
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
                        frame_rate = float(data[0].split(":")[1])
                        total_frame = float(data[1].split(":")[1])
                        total_time = float(data[2].split(":")[1])

                    video_x = self.read_video(frame_path, cur_fps=frame_rate, target_fps=30)
                    ecg_signals = self.resample_ecg(ecg, target_len=len(video_x))

                    sample = {}
                    sample["video_data"] = video_x.transpose(3, 0, 1, 2) # (C, T, H, W)
                    sample["ecg_data"] = ecg_signals
                    print(f'\rsaving {v_root}/{source_name}/sample.hdf5', end='')
                    with h5py.File(f'{v_root}/{source_name}/sample.hdf5', 'w') as f:
                        for key in sample.keys():
                            f.create_dataset(key, data=sample[key])

    def preprocess_UBFC(self):
        subject_list = os.listdir(self.dataset_dir)
        subject_list.remove('subject11')    # exist error hr (eq 0) in sample
        subject_list.remove('subject18')    # exist error hr (eq 0) in sample
        subject_list.remove('subject20')    # exist error hr (eq 0) in sample
        subject_list.remove('subject24')    # exist error hr (eq 0) in sample
        subject_list.sort()
        for subject in subject_list:
            pic_type = 'align_crop_pic'
            file_dir = os.path.join(self.dataset_dir, subject)
            video_dir = os.path.join(file_dir, pic_type)
            frame_list = os.listdir(video_dir)
            frame_list_int = [int(i.split(".")[0]) for i in frame_list]
            frame_list_int.sort() 
            frame_path = []
            for frame_name in frame_list_int:
                frame_path.append(os.path.join(video_dir, f"{frame_name}.png"))
            # read label
            with open(os.path.join(file_dir, 'ground_truth.txt'), 'r') as f:
                data = f.readlines()
            data_timestamp = np.array([float(strr.replace('e','E')) for strr in list(data[2].split())]) # 每一帧的时间戳，单位s, 起始的时间戳为0
            data_Hr = np.array([float(strr.replace('e','E')) for strr in list(data[1].split())])
            data_ecg = np.array([float(strr.replace('e','E')) for strr in list(data[0].split())])
            assert len(data_timestamp) == len(data_Hr) == len(data_ecg)
            assert len(data_timestamp) == len(frame_path)
            time_diffs = np.diff(data_timestamp)
            frame_rate = 1 / time_diffs.mean()

            video_x = self.read_video(frame_path, cur_fps=frame_rate, target_fps=30)
            ecg_signals = self.resample_ecg(data_ecg, target_len=len(video_x))
            
            sample = {}
            sample["video_data"] = video_x.transpose(3, 0, 1, 2) # (C, T, H, W)
            sample["ecg_data"] = ecg_signals
            print(f'\rsaving {file_dir}/sample.hdf5', end='')
            with h5py.File(f'{file_dir}/sample.hdf5', 'w') as f:
                for key in sample.keys():
                    f.create_dataset(key, data=sample[key])

    def preprocess_PURE(self):
        date_list = os.listdir(self.dataset_dir)
        date_list.sort()
        for date in date_list:
            # read video
            pic_type = 'align_crop_pic'
            video_dir = os.path.join(self.dataset_dir, date, pic_type)
            # read label
            json_file = os.path.join(self.dataset_dir, date, date + ".json")
            with open(json_file, 'r') as f:
                data = json.load(f)
            ecg_time_stamp = np.array([i['Timestamp'] for i in data['/FullPackage']])
            ecg = np.array([i['Value']['waveform'] for i in data['/FullPackage']])
            video_time_stamp = np.array([i['Timestamp'] for i in data['/Image']])
            # 基于json文件中的时间戳，保证视频图片文件名的顺序正确
            frame_path = []
            for i in range(len(video_time_stamp)):
                frame_path.append(os.path.join(video_dir, f"Image{video_time_stamp[i]}.png"))

            ecg_time_diffs = np.diff(ecg_time_stamp / 1e9)
            ecg_rate = 1 / ecg_time_diffs.mean()
            frame_time_diffs = np.diff(video_time_stamp / 1e9)
            frame_rate = 1 / frame_time_diffs.mean()

            video_x = self.read_video(frame_path, cur_fps=frame_rate, target_fps=30)
            ecg_signals = self.resample_ecg(ecg, target_len=len(video_x))

            sample = {}
            sample["video_data"] = video_x.transpose(3, 0, 1, 2) # (C, T, H, W)
            sample["ecg_data"] = ecg_signals
            print(f'\rsaving {self.dataset_dir}/{date}/sample.hdf5', end='')
            with h5py.File(f'{self.dataset_dir}/{date}/sample.hdf5', 'w') as f:
                for key in sample.keys():
                    f.create_dataset(key, data=sample[key])

    def preprocess_COHFACE(self):
        for px in sorted(os.listdir(self.dataset_dir)):
            px_path = os.path.join(self.dataset_dir, px)
            if not os.path.isdir(px_path) or px == 'protocols':
                continue
            for v_src in sorted(os.listdir(px_path)):
                if (px, v_src) in [('11', '2'), ('3', '2'), ('3', '3'), ('6', '2')]:
                    continue
                frame_dir = os.path.join(px_path, v_src, 'align_crop_pic')
                frame_path = []
                frame_list = os.listdir(frame_dir)
                frame_list_int = [int(i.split(".")[0]) for i in frame_list]
                frame_list_int.sort()
                for frame_name in frame_list_int:
                    frame_path.append(os.path.join(frame_dir, f"{frame_name:0>5}.png"))
                # read label
                with h5py.File(os.path.join(px_path, v_src, 'data.hdf5'), 'r') as f:
                    ecg = np.array(f['pulse'], dtype=np.float32)
                    time = np.array(f['time'], dtype=np.float32)

                frame_rate = len(frame_path) / time[-1]
                video_x = self.read_video(frame_path, cur_fps=frame_rate, target_fps=30)
                ecg_signals = self.resample_ecg(ecg, target_len=len(video_x))

                sample = {}
                sample["video_data"] = video_x.transpose(3, 0, 1, 2)[:, 10:, :, :] # (C, T, H, W)
                sample["ecg_data"] = ecg_signals[10:]   # remove the first 10 frames, COHFACE exists zero wave in the beginning
                print(f'\rsaving {px_path}/{v_src}/sample.hdf5', end='')
                with h5py.File(f'{px_path}/{v_src}/sample.hdf5', 'w') as f:
                    for key in sample.keys():
                        f.create_dataset(key, data=sample[key])

                

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, default='')
    args.add_argument('--dataset_dir', type=str, default='')
    args.add_argument('--h', type=int, default=64)
    args.add_argument('--w', type=int, default=64)
    args = args.parse_args()

    preprocess = PreprocessRppgDatasets(args)
    preprocess.preprocess()