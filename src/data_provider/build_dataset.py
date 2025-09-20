import torch
from torch.utils.data import Dataset
import pickle
import os

class HVACDataset(Dataset):
    """공조 시스템 시계열 데이터셋"""
    
    def __init__(self, data, seq_len, pred_len, feature_indices=None):
        """
        Args:
            data (numpy.ndarray): 전체 데이터 (n_samples, n_features)
            seq_len (int): 입력 시퀀스 길이
            pred_len (int): 예측 시퀀스 길이
            feature_indices (dict, optional): feature 타입별 인덱스
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_indices = feature_indices
        
        # 시계열 데이터에서 가능한 샘플 수 계산
        self.total_len = seq_len + pred_len
        
    def __len__(self):
        # 시퀀스를 만들 수 있는 샘플 수
        return len(self.data) - self.total_len + 1
    
    def __getitem__(self, idx):
        """
        Returns:
            x_seq: 입력 시퀀스 (seq_len, n_features)
            y_seq: 타겟 시퀀스 (pred_len, n_features)
        """
        # 입력 시퀀스
        x_begin = idx
        x_end = x_begin + self.seq_len
        x_seq = self.data[x_begin:x_end]
        
        # 타겟 시퀀스 (바로 다음 시점부터)
        y_begin = x_end
        y_end = y_begin + self.pred_len
        y_seq = self.data[y_begin:y_end]
        
        # Tensor로 변환
        x_seq = torch.FloatTensor(x_seq)
        y_seq = torch.FloatTensor(y_seq)
        
        return x_seq, y_seq
    
    def get_feature_info(self):
        """Feature 정보 반환"""
        return self.feature_indices