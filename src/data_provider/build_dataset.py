from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import warnings

warnings.filterwarnings('ignore')

class LG_BuildDataset(Dataset):
    def __init__(self, data_list: list):
        
        self.input_category = data_list[0]
        self.input_continuous = data_list[1]
        self.input_tsa = data_list[2]
        self.input_tsb = data_list[3]
        self.target = data_list[4]
        
        self.data_tsa = pd.DataFrame(self.input_tsa,
                                    columns=self.input_tsa.columns,
                                    index=self.input_tsa.index)
        
        self.data_tsb = pd.DataFrame(self.input_tsb,
                                    columns=self.input_tsb.columns,
                                    index=self.input_tsb.index)
        
        self.valid_idxs = np.array(self.input_category.index)
        
        print(f"# of valid windows: {len(self.valid_idxs)}")

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        # Tabular dataset
        input_tabular_cat = np.array(self.input_category.loc[i])
        input_tabular_con = np.array(self.input_continuous.loc[i])
        
        # Time-series A dataset
        input_tsa = np.array(self.input_tsa.loc[i])

        # Time-series B dataset
        input_tsb = np.array(self.input_tsb.loc[i])

        # Label dataset
        target = np.array(self.target.loc[i])

        item = {
            'inputs': {
                'tabular': {
                    'cat': torch.Tensor(input_tabular_cat),
                    'con': torch.Tensor(input_tabular_con)
                },
                'tsa': torch.Tensor(input_tsa),
                'tsb': torch.Tensor(input_tsb)
            },
            'targets': torch.Tensor(target)
        }
                
        return item
    
    def __len__(self):        
        return len(self.valid_idxs)
    
    