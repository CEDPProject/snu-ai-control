from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pdb
import warnings
from sklearn.model_selection import train_test_split
from collections import Counter

warnings.filterwarnings('ignore')

def load_dataset(
                category_name: str,
                datadir: str,
                scale: str,
                train_only: bool = False,
                pre_training: bool = False,
                data_num: int = 0,
                split_rate: list = [0.8, 0.1, 0.1]):
    
    # 태양광 발전량
    if category_name == 'solar':
        location     = pd.read_csv(os.path.join(datadir, "태양광/샘플_태양광발전단지_위치정보.csv"), encoding='cp949')
        ecmwf_fcst   = pd.read_csv(os.path.join(datadir, "태양광/샘플_태양광발전단지_ECMWF_수치예보데이터.csv"))
        um_fcst      = pd.read_csv(os.path.join(datadir, "태양광/샘플_태양광발전단지_UM_수치예보데이터.csv"))
        wrf_fcst     = pd.read_csv(os.path.join(datadir, "태양광/샘플_태양광발전단지_WRF_수치예보데이터.csv"))
    # 풍력 발전량
    elif category_name == 'wind':
        location     = pd.read_csv(os.path.join(datadir, "풍력/샘플_풍력발전단지_호기별_위치정보.csv"), encoding='euc-kr')
        ecmwf_fcst   = pd.read_csv(os.path.join(datadir, "풍력/샘플_풍력발전단지_ECMWF_수치예보데이터.csv"))
        um_fcst      = pd.read_csv(os.path.join(datadir, "풍력/샘플_풍력발전단지_UM_수치예보데이터.csv"))
        wrf_fcst     = pd.read_csv(os.path.join(datadir, "풍력/샘플_풍력발전단지_WRF_수치예보데이터.csv"), low_memory=False, encoding='euc-kr')

    # 실적 데이터
    weather_gt = pd.read_csv(os.path.join(datadir, "샘플_풍력발전단지_호기별_위치정보.csv"), encoding='euc-kr')
    power_gt  = pd.read_csv(os.path.join(datadir, "샘플_풍력발전단지_호기별_위치정보.csv"), encoding='euc-kr')

    if not train_test_predefined:
        df_raw_tabular = pd.read_csv(os.path.join(datadir, 'tabular_data.csv'), index_col=0)
        df_raw_tsa = pd.read_csv(os.path.join(datadir, 'SA_data.csv'), index_col=0)
        df_raw_tsb = pd.read_csv(os.path.join(datadir, 'SB_data.csv'), index_col=0)
        df_raw_target = pd.read_csv(os.path.join(datadir, 'target_data.csv'), index_col=0)
    else:
        if pre_training:
            data_num=f'Total_Equipment'
        else:
            data_num=f'Equipment_{data_num}'
        df_raw_tabular_train = pd.read_csv(os.path.join(datadir, data_num,'tabular_train.csv'), index_col=0)
        df_raw_tabular_test = pd.read_csv(os.path.join(datadir, data_num,'tabular_test.csv'), index_col=0)
        df_raw_tsa_train = pd.read_csv(os.path.join(datadir, data_num,'SA_train.csv'), index_col=0)
        df_raw_tsa_test = pd.read_csv(os.path.join(datadir, data_num,'SA_test.csv'), index_col=0)
        df_raw_tsb_train = pd.read_csv(os.path.join(datadir, data_num,'SB_train.csv'), index_col=0)
        df_raw_tsb_test = pd.read_csv(os.path.join(datadir, data_num,'SB_test.csv'), index_col=0)
        df_raw_target_train = pd.read_csv(os.path.join(datadir, data_num,'label_train.csv'), index_col=0)
        df_raw_target_test = pd.read_csv(os.path.join(datadir, data_num,'label_test.csv'), index_col=0)

    train_idx = df_raw_tabular_train.index
    _, _, train_idx, valid_idx = train_test_split(df_raw_tabular_train, train_idx, train_size = split_rate[0], shuffle=True, stratify=df_raw_tabular_train[['CAT_001','CAT_002']])

    # tabular data information
    columns_list = list(df_raw_tabular_train.columns)

    category_var = [cat for cat in columns_list if 'CAT' in cat]
    cat_field_dims = list(df_raw_tabular_train[category_var].nunique())
    cat_embed_input = [(u,i) for u,i in zip(category_var, cat_field_dims)]
    num_category = len(category_var)
    
    continuous_var = [con for con in columns_list if 'FLT' in con or 'INT' in con]
    column_idx = {k:v for v,k in enumerate(columns_list)}
    num_continuous = len(continuous_var)
    
    tabular_category_train = df_raw_tabular_train[category_var]
    tabular_continuous_train = df_raw_tabular_train[continuous_var]
    tabular_category_test = df_raw_tabular_test[category_var]
    tabular_continuous_test = df_raw_tabular_test[continuous_var]
    
    information_dict = {
        'cat_field_dims' : cat_field_dims,
        'cat_embed_input' : cat_embed_input,
        'column_idx' : column_idx,
        'num_category' : num_category,
        'num_continuous' : num_continuous
    }
    
    
    # category
    trn_category = tabular_category_train.loc[train_idx]
    val_category = tabular_category_train.loc[valid_idx]
    tst_category = tabular_category_test

    # continuous
    trn_continuous = tabular_continuous_train.loc[train_idx]
    val_continuous = tabular_continuous_train.loc[valid_idx]
    tst_continuous = tabular_continuous_test

    # tsa
    trn_tsa = df_raw_tsa_train.loc[train_idx]
    val_tsa = df_raw_tsa_train.loc[valid_idx]
    tst_tsa = df_raw_tsa_test

    # tsb
    trn_tsb = df_raw_tsb_train.loc[train_idx]
    val_tsb = df_raw_tsb_train.loc[valid_idx]
    tst_tsb = df_raw_tsb_test

    # target
    trn_target = df_raw_target_train.loc[train_idx]
    val_target = df_raw_target_train.loc[valid_idx]
    tst_target = df_raw_target_test
        
    if scale is not None:
        if scale == 'minmax':
            scaler_con = MinMaxScaler(feature_range=(-1, 1))
            scaler_tsa = MinMaxScaler(feature_range=(-1, 1))
            scaler_tsb = MinMaxScaler(feature_range=(-1, 1))
            
            scaler_con.fit(trn_continuous)
            scaler_tsa.fit(trn_tsa)
            scaler_tsb.fit(trn_tsb)
            
            trn_scaling_continuous = scaler_con.transform(trn_continuous)
            val_scaling_continuous = scaler_con.transform(val_continuous)
            tst_scaling_continuous = scaler_con.transform(tst_continuous)

            trn_scaling_tsa = scaler_tsa.transform(trn_tsa)
            val_scaling_tsa = scaler_tsa.transform(val_tsa)
            tst_scaling_tsa = scaler_tsa.transform(tst_tsa)
            
            trn_scaling_tsb = scaler_tsb.transform(trn_tsb)
            val_scaling_tsb = scaler_tsb.transform(val_tsb)
            tst_scaling_tsb = scaler_tsb.transform(tst_tsb)

        elif scale == 'minmax_square':
            scaler_con = MinMaxScaler()
            scaler_tsa = MinMaxScaler()
            scaler_tsb = MinMaxScaler()
            
            scaler_con.fit(trn_continuous)
            scaler_tsa.fit(trn_tsa)
            scaler_tsb.fit(trn_tsb)
            
            trn_scaling_continuous = scaler_con.transform(trn_continuous) ** 2
            val_scaling_continuous = scaler_con.transform(val_continuous) ** 2
            tst_scaling_continuous = scaler_con.transform(tst_continuous) ** 2
            
            trn_scaling_tsa = scaler_tsa.transform(trn_tsa) ** 2
            val_scaling_tsa = scaler_tsa.transform(val_tsa) ** 2
            tst_scaling_tsa = scaler_tsa.transform(tst_tsa) ** 2
            
            trn_scaling_tsb = scaler_tsb.transform(trn_tsb) ** 2
            val_scaling_tsb = scaler_tsb.transform(val_tsb) ** 2
            tst_scaling_tsb = scaler_tsb.transform(tst_tsb) ** 2

        elif scale == 'standard':
            scaler_con = StandardScaler()
            scaler_tsa = StandardScaler()
            scaler_tsb = StandardScaler()
            
            scaler_con.fit(trn_continuous)
            scaler_tsa.fit(trn_tsa)
            scaler_tsb.fit(trn_tsb)
            
            trn_scaling_continuous = scaler_con.transform(trn_continuous)
            val_scaling_continuous = scaler_con.transform(val_continuous)
            tst_scaling_continuous = scaler_con.transform(tst_continuous)
            
            trn_scaling_tsa = scaler_tsa.transform(trn_tsa)
            val_scaling_tsa = scaler_tsa.transform(val_tsa)
            tst_scaling_tsa = scaler_tsa.transform(tst_tsa)
            
            trn_scaling_tsb = scaler_tsb.transform(trn_tsb)
            val_scaling_tsb = scaler_tsb.transform(val_tsb)
            tst_scaling_tsb = scaler_tsb.transform(tst_tsb)
        
        # scaled continuous
        trn_continuous = pd.DataFrame(trn_scaling_continuous,
                                    columns=trn_continuous.columns,
                                    index=trn_continuous.index)

        val_continuous = pd.DataFrame(val_scaling_continuous,
                                    columns=val_continuous.columns,
                                    index=val_continuous.index)
        
        tst_continuous = pd.DataFrame(tst_scaling_continuous,
                                    columns=tst_continuous.columns,
                                    index=tst_continuous.index)

        # scaled tsa
        trn_tsa = pd.DataFrame(trn_scaling_tsa,
                                    columns=trn_tsa.columns,
                                    index=trn_tsa.index)

        val_tsa = pd.DataFrame(val_scaling_tsa,
                                    columns=val_tsa.columns,
                                    index=val_tsa.index)
        
        tst_tsa = pd.DataFrame(tst_scaling_tsa,
                                    columns=tst_tsa.columns,
                                    index=tst_tsa.index)

        # scaled tsb
        trn_tsb = pd.DataFrame(trn_scaling_tsb,
                                    columns=trn_tsb.columns,
                                    index=trn_tsb.index)

        val_tsb = pd.DataFrame(val_scaling_tsb,
                                    columns=val_tsb.columns,
                                    index=val_tsb.index)
        
        tst_tsb = pd.DataFrame(tst_scaling_tsb,
                                    columns=tst_tsb.columns,
                                    index=tst_tsb.index)
        
        print(f'{scale} Normalization done')
    
    trn_list = [trn_category, trn_continuous, trn_tsa, trn_tsb, trn_target]
    val_list = [val_category, val_continuous, val_tsa, val_tsb, val_target]
    tst_list = [tst_category, tst_continuous, tst_tsa, tst_tsb, tst_target]
    
    print(f"train: CAT_001: {Counter(trn_category['CAT_001'])}  CAT_002: {Counter(trn_category['CAT_002'])}")
    print(f"valid: CAT_001: {Counter(val_category['CAT_001'])}  CAT_002: {Counter(val_category['CAT_002'])}")
    print(f"test: CAT_001: {Counter(tst_category['CAT_001'])}  CAT_002: {Counter(tst_category['CAT_002'])}")
    
    return information_dict, trn_list, val_list, tst_list


# 발전 호기 필터링
def filtering_data(fcst, location, category_name):
    if category_name == 'solar':
        unit_ids = location['GEN_ID'].unique()
    elif category_name == 'wind':
        unit_ids = location['UNIT_ID'].unique()
    filtered_wind_ecmwf = fcst[fcst['AREA_GRP_ID'].isin(unit_ids)]
    return filtered_wind_ecmwf