import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob

class UrbanSoundDataset(Dataset):
    def __init__(self, root_dir, target_shape=(40, 1501)):
        """
        Args:
            root_dir (str): 데이터가 저장된 루트 디렉터리 경로
            target_shape (tuple): 목표 데이터 크기 (frames, coefficients)
        """
        self.data = []
        self.labels = []
        self.target_shape = target_shape
        self.load_data(root_dir)

    def load_data(self, root_dir):
        for label_folder in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_folder)
            if os.path.isdir(label_path):
                for npy_file in glob(os.path.join(label_path, '*.npy')):
                    data = np.load(npy_file)
                    data = self.pad_to_target(data, self.target_shape)  # 패딩 추가
                    self.data.append(data)
                    self.labels.append(int(label_folder))

    def pad_to_target(self, data, target_shape):
        """데이터를 목표 크기로 패딩합니다."""
        if data.shape == target_shape:
            return data
        elif data.shape[1] < target_shape[1]:  # 열이 부족한 경우
            padding = ((0, 0), (0, target_shape[1] - data.shape[1]))
            data = np.pad(data, padding, mode='constant')
        else:  # 열이 초과하는 경우, 목표 길이로 자르기
            data = data[:, :target_shape[1]]
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label
