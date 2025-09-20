import torch
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader
from data_provider.load_dataset import load_dataset 
from data_provider.build_dataset import HVACDataset 
from omegaconf import OmegaConf

warnings.filterwarnings('ignore')

def create_dataloader(
                datadir: str,
                scaler: str,
                batch_size: int,
                shuffle: bool,
                num_workers: int,
                pin_memory: bool,
                drop_last: bool,
                data_version: str,
                model_name: str,
                environment_name: str,
                ord_categorical_feature: list,
                categorical_feature: list,
                numerical_feature: list,
                del_feature: list = None,
                split_rate: list = [0.7, 0.1, 0.2],
                seq_len: int = 96,
                pred_len: int = 96,
                split_date: str = None,
                end_date: str = None,
                start_date: str = None,
                ):

    assert scaler in (None, 'minmax', 'minmax_square', 'standard')

    data_dict, features, scaler_dict = load_dataset(
                            datadir                 = datadir,
                            scaler                  = scaler,
                            ord_categorical_feature = ord_categorical_feature,
                            categorical_feature     = categorical_feature,
                            numerical_feature       = numerical_feature,
                            del_feature             = del_feature,
                            data_version            = data_version,
                            split_rate              = split_rate,
                            split_date              = split_date,
                            end_date                = end_date,
                            start_date              = start_date,
                            environment_name        = environment_name
                            )
    
    trn_dataset = HVACDataset(data          = data_dict['train'],
                            seq_len         = seq_len, 
                            pred_len        = pred_len,
                            feature_indices = scaler_dict.get('feature_indices')
                            )
    
    val_dataset = HVACDataset(data          = data_dict['val'],
                            seq_len         = seq_len, 
                            pred_len        = pred_len,
                            feature_indices = scaler_dict.get('feature_indices')
                            )
    
    tst_dataset = HVACDataset(data          = data_dict['test'],
                            seq_len         = seq_len, 
                            pred_len        = pred_len,
                            feature_indices = scaler_dict.get('feature_indices')
                            )
    
    # torch dataloader
    trn_dataloader = DataLoader(trn_dataset,
                                batch_size     =  batch_size,
                                shuffle        =  shuffle,
                                num_workers    =  num_workers,
                                pin_memory     =  pin_memory,
                                drop_last      =  drop_last)

    val_dataloader = DataLoader(val_dataset,
                                batch_size     =  batch_size,
                                shuffle        =  shuffle,
                                num_workers    =  num_workers,
                                pin_memory     =  pin_memory,
                                drop_last      =  drop_last)

    tst_dataloader = DataLoader(tst_dataset,
                                batch_size     =  batch_size,
                                shuffle        =  False,
                                num_workers    =  num_workers,
                                pin_memory     =  pin_memory,
                                drop_last      =  False)
    
    # 메타데이터 구성
    metadata = {
        'features': features,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'data_shapes': {k: v.shape if v is not None else None 
                        for k, v in data_dict.items()},
        'n_features': len(features),
        'feature_types': {
            'ordinal': ord_categorical_feature,
            'categorical': categorical_feature,
            'numerical': numerical_feature
        },
        'batch_size': batch_size,
        'model_name': model_name,
        'environment_name': environment_name,
        'data_version': data_version
    }
    
    if 'MPC' in model_name:
        return data_dict, metadata, scaler_dict
    else:
        return (trn_dataloader, val_dataloader, tst_dataloader), metadata, scaler_dict
