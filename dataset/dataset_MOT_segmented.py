import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from glob import glob


class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, mode = 'train'):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        
        self.mcs_sessions = ["349e4383-da38-4138-8371-9a5fed63a56a","015b7571-9f0b-4db4-a854-68e57640640d","c613945f-1570-4011-93a4-8c8c6408e2cf","dfda5c67-a512-4ca2-a4b3-6a7e22599732","7562e3c0-dea8-46f8-bc8b-ed9d0f002a77","275561c0-5d50-4675-9df1-733390cd572f","0e10a4e3-a93f-4b4d-9519-d9287d1d74eb","a5e5d4cd-524c-4905-af85-99678e1239c8","dd215900-9827-4ae6-a07d-543b8648b1da","3d1207bf-192b-486a-b509-d11ca90851d7","c28e768f-6e2b-4726-8919-c05b0af61e4a","fb6e8f87-a1cc-48b4-8217-4e8b160602bf","e6b10bbf-4e00-4ac0-aade-68bc1447de3e","d66330dc-7884-4915-9dbb-0520932294c4","0d9e84e9-57a4-4534-aee2-0d0e8d1e7c45","2345d831-6038-412e-84a9-971bc04da597","0a959024-3371-478a-96da-bf17b1da15a9","ef656fe8-27e7-428a-84a9-deb868da053d","c08f1d89-c843-4878-8406-b6f9798a558e","d2020b0e-6d41-4759-87f0-5c158f6ab86a","8dc21218-8338-4fd4-8164-f6f122dc33d9"]
        self.mcs_scores = [4,4,2,3,2,4,3,3,2,3,4,3,4,2,2,3,4,4,3,3,3]

        if dataset_name == 'mcs':
            self.data_root = '/home/ubuntu/data/HumanML3D' + "/" + mode
            self.max_motion_length = 196 # Maximum length of sequence, change this for different lengths
            self.meta_dir = '/home/ubuntu/data/HumanML3D'
        
        self.file_list = glob("/home/ubuntu/data/MCS_DATA/Data/*/OpenSimData/Dynamics/*_segment_*/kinematics_*_muscle_driven.mot")
        self.train_data = []
        self.train_lengths = []
        self.train_names = []
        
        self.test_data = []
        self.test_lengths = []
        self.test_names = []
        
        self.mode = mode
        
        for file in tqdm(self.file_list):
            tmp_name = file.split('/')[-1]
            if "sqt" not in tmp_name and "SQT" not in tmp_name and "Sqt" not in tmp_name:
                continue
            with open(file,'r') as f:
                file_data = f.read().split('\n')
                # print(file_data)
                data = {'info':'', 'poses': []}
                read_header = False
                read_rows = 0
                
                for line in file_data:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    
                    if not read_header:
                        if line == 'endheader':
                            read_header = True
                            continue
                        if '=' not in line:
                            data['info'] += line + '\n'
                        else:
                            k,v = line.split('=')
                            if v.isnumeric():
                                data[k] = int(v)
                            else:
                                data[k] = v
                    else:
                        rows = line.split()
                        if read_rows == 0:
                            data['headers'] = rows
                        else:
                            rows = [float(row) for row in rows]
                            data['poses'].append(rows)
                        read_rows += 1
            if data['nRows'] < self.window_size:
                continue
            data['poses'] = np.array(data['poses'])[:,1:34] # Change to remove time 
            # print(data['poses'].shape)
            z = file.split('/')[6]
            if z not in self.mcs_sessions:
                self.train_data.append(data['poses'])
                self.train_lengths.append(data['nRows'])
                self.train_names.append(file)
            else:
                self.test_data.append(data['poses'])
                self.test_lengths.append(data['nRows'])
                self.test_names.append(file)
        
        print("Total number of train motions {}".format(len(self.train_data)))
        print("Total number of test motions {}".format(len(self.test_data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.train_lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_data)
        if self.mode == 'test':
            return len(self.test_data)

    def __getitem__(self, item):
        if self.mode == 'train':
            motion = self.train_data[item]
            len_motion = len(motion) if len(motion) <=self.max_motion_length else self.max_motion_length
            name = self.train_names[item]
            
            # idx = random.randint(0, len(motion) - self.window_size)
            # motion = motion[idx:idx+self.window_size]
            
            if len(motion) >= self.max_motion_length:
                idx = random.randint(0, len(motion) - self.max_motion_length)
                motion = motion[idx:idx+self.max_motion_length]
            else:
                # Pad with 0
                # pad_width = self.max_motion_length - len(motion)
                # motion = np.pad(motion, ((0, pad_width), (0,0)), mode='constant', constant_values=0)
                
                # Repeat motion from start
                repeat_count = (self.max_motion_length + len(motion) - 1) // len(motion)  # Calculate repetitions needed
                motion = np.tile(motion, (repeat_count, 1))[:self.max_motion_length]
            
            "Z Normalization"
            # motion = (motion - self.mean) / self.std

            return motion, len_motion, name
        
        if self.mode == 'limo':
            motion = self.data[item]
            len_motion = len(motion)
            name = self.names[item]
            
            if len_motion < self.max_motion_length:
                pad_width = self.max_motion_length - len_motion
                motion = np.pad(motion, ((0, pad_width), (0,0)), mode='constant', constant_values=0)
                return [motion], [len_motion], [name]
            
            subsequences = []
            subsequence_lengths = []
            names = []

            for start_idx in range(0, len_motion - self.max_motion_length + 1,4):
                subseq = motion[start_idx:start_idx + self.max_motion_length]
                subsequences.append(subseq)
                subsequence_lengths.append(self.max_motion_length)
                names.append(name)

            return subsequences, subsequence_lengths, names

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4,
               mode = 'train'):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length, mode=mode)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
