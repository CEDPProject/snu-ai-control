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
                target_features: list,
                input_features: list,
                ord_categorical_feature: list,
                categorical_feature: list,
                cat_encode_method: str,
                numerical_feature: list,
                del_feature: list = None,
                split_rate: list = [0.7, 0.1, 0.2],
                seq_len: int = 96,
                stride_len: int = 1,
                pred_len: int = 96,
                split_date: str = None,
                end_date: str = None,
                start_date: str = None,
                ):

    assert scaler in (None, 'minmax', 'minmax_square', 'standard')
    
    all_categorical_feature = ord_categorical_feature + categorical_feature

    data_dict, features, scaler_dict = load_dataset(
                            datadir                 = datadir,
                            scaler                  = scaler,
                            ord_categorical_feature = ord_categorical_feature,
                            categorical_feature     = categorical_feature,
                            cat_encode_method       = cat_encode_method,
                            numerical_feature       = numerical_feature,
                            target_features         = target_features,
                            del_feature             = del_feature,
                            data_version            = data_version,
                            split_rate              = split_rate,
                            split_date              = split_date,
                            end_date                = end_date,
                            start_date              = start_date,
                            environment_name        = environment_name
                            )
    
    trn_dataset = HVACDataset(data               = data_dict['train']['data'],
                            timestamps           = data_dict['train']['timestamps'],
                            seq_len              = seq_len, 
                            stride_len           = stride_len,
                            pred_len             = pred_len,
                            features             = features,
                            target_features      = target_features,
                            input_features       = input_features,
                            categorical_feature  = all_categorical_feature,
                            numerical_feature    = numerical_feature,
                            feature_indices      = scaler_dict.get('feature_indices')
                            )
    
    val_dataset = HVACDataset(data               = data_dict['val']['data'],
                            timestamps           = data_dict['val']['timestamps'],
                            seq_len              = seq_len, 
                            stride_len           = stride_len,
                            pred_len             = pred_len,
                            features             = features,
                            target_features      = target_features,
                            input_features       = input_features,
                            categorical_feature  = all_categorical_feature,
                            numerical_feature    = numerical_feature,
                            feature_indices      = scaler_dict.get('feature_indices')
                            )
    
    tst_dataset = HVACDataset(data               = data_dict['test']['data'],
                            timestamps           = data_dict['test']['timestamps'],
                            seq_len              = seq_len, 
                            stride_len           = stride_len,
                            pred_len             = pred_len,
                            features             = features,
                            target_features      = target_features,
                            input_features       = input_features,
                            categorical_feature  = all_categorical_feature,
                            numerical_feature    = numerical_feature,
                            feature_indices      = scaler_dict.get('feature_indices')
                            )
    
    # torch dataloader
    trn_dataloader = DataLoader(trn_dataset,
                                batch_size     =  batch_size,
                                shuffle        =  shuffle,
                                collate_fn     =  custom_collate_fn,
                                num_workers    =  num_workers,
                                pin_memory     =  pin_memory,
                                drop_last      =  drop_last)

    val_dataloader = DataLoader(val_dataset,
                                batch_size     =  batch_size,
                                shuffle        =  shuffle,
                                collate_fn     =  custom_collate_fn,
                                num_workers    =  num_workers,
                                pin_memory     =  pin_memory,
                                drop_last      =  drop_last)

    tst_dataloader = DataLoader(tst_dataset,
                                batch_size     =  batch_size,
                                shuffle        =  False,
                                collate_fn     =  custom_collate_fn,
                                num_workers    =  num_workers,
                                pin_memory     =  pin_memory,
                                drop_last      =  False)
    
    # 메타데이터 구성
    metadata = {
        'features': features,
        'seq_len': seq_len,
        'pred_len': pred_len,
        'n_features': len(features),
        'feature_types': {
            'ordinal': ord_categorical_feature,
            'categorical': categorical_feature,
            'numerical': numerical_feature,
            'n_num': len(numerical_feature),
            'n_cat': len(all_categorical_feature),
            'n_target': len(target_features),
        },
        'batch_size': batch_size,
        'model_name': model_name,
        'environment_name': environment_name,
        'data_version': data_version
    }
    
    if 'MPC' in model_name:
        return trn_dataloader, val_dataloader, tst_dataloader, metadata, scaler_dict
    else:
        return trn_dataloader, val_dataloader, tst_dataloader, metadata, scaler_dict

def custom_collate_fn(batch):
    """
    Timestamps는 tensor 변환하지 않고 list로 유지
    """
    input_cat = [item['input']['cat'] for item in batch]
    input_num = [item['input']['num'] for item in batch]
    input_ts = [item['input']['ts'] for item in batch]  # list로 유지
    
    target_values = [item['target']['target'] for item in batch]
    target_ts = [item['target']['ts'] for item in batch]  # list로 유지
    
    # None이 아닌 것들만 stack
    input_cat_batch = torch.stack(input_cat) if input_cat[0] is not None else None
    input_num_batch = torch.stack(input_num) if input_num[0] is not None else None
    target_batch = torch.stack(target_values)
    
    return {
        "input": {
            "cat": input_cat_batch,
            "num": input_num_batch,
            "ts": input_ts  # list of numpy arrays
        },
        "target": {
            "target": target_batch,
            "ts": target_ts  # list of numpy arrays
        }
    }