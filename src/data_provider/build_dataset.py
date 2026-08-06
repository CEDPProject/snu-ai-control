import numpy as np
import torch
from torch.utils.data import Dataset

class HVACDataset(Dataset):
    def __init__(
        self,
        data,
        timestamps,
        seq_len: int,
        pred_len: int,
        stride_len: int = 1,
        features: list = None,
        target_features: list = None,
        input_features: list = None,
        categorical_feature: list = None,
        numerical_feature: list = None,
        feature_indices: dict = None,
    ):
        self.data = data
        self.timestamps = timestamps  # 추가
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride_len = max(1, int(stride_len))

        # 이름 표준화
        self.features = [str(f).strip() for f in features]
        self.feature_indices = feature_indices
        self.categorical_feature = list(categorical_feature) if categorical_feature else []
        self.numerical_feature   = list(numerical_feature)   if numerical_feature   else []

        name_to_idx = {name: i for i, name in enumerate(self.features)}

        # target index
        def to_indices_exact(names, label):
            idxs, missing = [], []
            for it in names:
                if isinstance(it, int):
                    idx = int(it)
                    if 0 <= idx < len(self.features):
                        idxs.append(idx)
                    else:
                        missing.append(it)
                else:
                    key = str(it).strip()
                    if key in name_to_idx:
                        idxs.append(name_to_idx[key])
                    else:
                        missing.append(it)
            if missing:
                raise ValueError(f"{label}에서 못 찾은 열: {missing}")
            return list(dict.fromkeys(idxs))

        self.target_idx = to_indices_exact(target_features, "타깃 feature")
        self.target_names = [self.features[i] for i in self.target_idx]
        
        # input features: 기본은 전체
        if input_features is None:
            self.input_idx = list(range(len(self.features)))
            input_feature_names = self.features
        else:
            self.input_idx = to_indices_exact(input_features, "입력 feature")
            input_feature_names = [self.features[i] for i in self.input_idx]

        # === Label encoding: 컬럼명이 그대로 유지되므로 단순 매칭 ===
        # 1) 범주형 열 이름 - exact match
        if self.categorical_feature:
            cat_names = [n for n in self.categorical_feature if n in input_feature_names]
        else:
            cat_names = []

        # 2) 수치형: 명시되지 않으면 입력 피처에서 범주형 이외
        if self.numerical_feature:
            numerical_names = [n for n in self.numerical_feature if n in input_feature_names]
        else:
            numerical_names = [n for n in input_feature_names if n not in cat_names]

        # 3) 이름 -> 인덱스
        self.input_cat_names = cat_names
        self.input_num_names = numerical_names
        self.input_cat_idx = [name_to_idx[n] for n in cat_names]
        self.input_num_idx = [name_to_idx[n] for n in numerical_names]
        
        # cat/num 중복 방지
        overlap = set(self.input_cat_idx) & set(self.input_num_idx)
        if overlap:
            self.input_num_idx = [i for i in self.input_num_idx if i not in overlap]

        self.input_cat_names = [self.features[i] for i in self.input_cat_idx]
        self.input_num_names = [self.features[i] for i in self.input_num_idx]
        
        # 한 샘플 길이/개수
        self.total_len = self.seq_len + self.pred_len
        max_start = len(self.data) - self.total_len
        self.n_samples = 0 if max_start < 0 else (max_start // self.stride_len) + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start   = idx * self.stride_len
        x_begin = start
        x_end   = x_begin + self.seq_len
        y_begin = x_end
        y_end   = y_begin + self.pred_len

        # pandas면 .iloc, 아니면 원본 그대로 쓰기
        data = self.data.iloc if hasattr(self.data, "iloc") else self.data
        ts   = (self.timestamps.iloc if hasattr(self.timestamps, "iloc") else self.timestamps) \
            if self.timestamps is not None else None
        
        # Input
        input_cat = None
        input_num = None
        input_ts  = ts[x_begin:x_end] if ts is not None else None

        if len(self.input_cat_idx) > 0:
            x_cat = data[x_begin:x_end, self.input_cat_idx]
            input_cat = torch.as_tensor(x_cat, dtype=torch.float32)

        if len(self.input_num_idx) > 0:
            x_num = data[x_begin:x_end, self.input_num_idx]
            input_num = torch.as_tensor(x_num, dtype=torch.float32)

        # Target
        y_seq = data[y_begin:y_end, self.target_idx]
        target_values = torch.as_tensor(y_seq, dtype=torch.float32)
        target_ts = ts[y_begin:y_end] if ts is not None else None

        return {
            "input":  {"cat": input_cat, "num": input_num, "ts": input_ts},
            "target": {"target": target_values, "ts": target_ts},
        }

    def get_feature_info(self):
        return {
            "all_features": self.features,
            "input_idx_all": getattr(self, "input_idx", None),
            "input_cat_idx": self.input_cat_idx,
            "input_num_idx": self.input_num_idx,
            "target_idx": self.target_idx,
            "input_categorical_features": self.input_cat_names,
            "input_numerical_features": self.input_num_names,
            "target_features": self.target_names,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "stride_len": self.stride_len,
            "n_samples": len(self),
            "feature_indices": self.feature_indices,
        }