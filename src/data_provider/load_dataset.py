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
                structure: str,
                train_only: bool = False,
                pre_training: bool = False,
                data_num: int = 0,
                del_feature: list = ['SWDDIR'],
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
    power_gt   = pd.read_csv(os.path.join(datadir, "샘플_풍력발전단지_호기별_위치정보.csv"), encoding='euc-kr')
    
    # 발전 호기 필터링
    ecmwf_fcst = filtering_id(ecmwf_fcst, location, category_name)
    um_fcst    = filtering_id(um_fcst, location, category_name)
    wrf_fcst   = filtering_id(wrf_fcst, location, category_name)

    ecmwf_fcst = remove_columns(ecmwf_fcst, del_feature)
    um_fcst = remove_columns(um_fcst, del_feature)
    wrf_fcst = remove_columns(wrf_fcst, del_feature)

    # 동일 CRTN_TM, FCST_TM 필터링
    ecmwf_fcst, um_fcst, wrf_fcst, align_time = align_times(ecmwf_fcst, um_fcst, wrf_fcst)
    
    # 데이터 Input 구조 
    if structure == 'independence':
        data_list = [ecmwf_fcst, um_fcst, wrf_fcst, weather_gt, power_gt]
    elif structure == 'merge':
        combined_df = pd.concat([ecmwf_fcst, um_fcst, wrf_fcst])
        combined_df = combined_df.sort_values(by=["AREA_GRP_ID", "CRTN_TM", "FCST_TM"]).reset_index(drop=True)
        data_list = [combined_df, location, weather_gt, power_gt]
    
    
    

    # 데이터 전처리
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
    print(f"train: CAT_001: {Counter(trn_category['CAT_001'])}  CAT_002: {Counter(trn_category['CAT_002'])}")
    print(f"valid: CAT_001: {Counter(val_category['CAT_001'])}  CAT_002: {Counter(val_category['CAT_002'])}")
    print(f"test: CAT_001: {Counter(tst_category['CAT_001'])}  CAT_002: {Counter(tst_category['CAT_002'])}")
    
    return information_dict, trn_list, val_list, tst_list


# 발전 호기 필터링
def filtering_id(fcst, location, category_name):
    if category_name == 'solar':
        unit_ids = location['GEN_ID'].unique()
    elif category_name == 'wind':
        unit_ids = location['UNIT_ID'].unique()
    filtered_wind_ecmwf = fcst[fcst['AREA_GRP_ID'].isin(unit_ids)]
    return filtered_wind_ecmwf

# 동일 CRTN_TM, FCST_TM 필터링
def align_times(df1, df2, df3):
    # 날짜 형식을 동일하게 맞춤
    for df in [df1, df2, df3]:
        df = df.copy()  # 원본 DataFrame을 보호하기 위해 복사본을 사용
        df.loc[:, 'CRTN_TM'] = pd.to_datetime(df['CRTN_TM'], format='%Y%m%d%H%M')
        df.loc[:, 'FCST_TM'] = pd.to_datetime(df['FCST_TM'], format='%Y%m%d%H%M')

    # 중복 제거 후 공통 시간을 찾기 위해 각 DataFrame에서 (CRTN_TM, FCST_TM) 쌍 추출
    df1_times = set(df1[['CRTN_TM', 'FCST_TM']].drop_duplicates().apply(tuple, axis=1))
    df2_times = set(df2[['CRTN_TM', 'FCST_TM']].drop_duplicates().apply(tuple, axis=1))
    df3_times = set(df3[['CRTN_TM', 'FCST_TM']].drop_duplicates().apply(tuple, axis=1))

    # 공통 시간 계산
    common_times = sorted(df1_times.intersection(df2_times, df3_times))
    
    # 공통 시간으로 각 데이터프레임 필터링
    filtered_df1 = df1[df1[['CRTN_TM', 'FCST_TM']].apply(tuple, axis=1).isin(common_times)]
    filtered_df2 = df2[df2[['CRTN_TM', 'FCST_TM']].apply(tuple, axis=1).isin(common_times)]
    filtered_df3 = df3[df3[['CRTN_TM', 'FCST_TM']].apply(tuple, axis=1).isin(common_times)]

    return filtered_df1, filtered_df2, filtered_df3, common_times

def remove_columns(df, columns):
    df = df.drop(columns=[col for col in columns if col in df.columns], errors='ignore')
    return df

def apply_scaling(data_list, scale, structure):
    scaler_map = {
        'minmax': MinMaxScaler(feature_range=(-1, 1)),
        'minmax_square': MinMaxScaler(),
        'standard': StandardScaler()
    }
    scaler = scaler_map.get(scale)
    scaler_dict = {}

    if structure == 'independence':
        scaled_data_list = []
        for i, df in enumerate(data_list):
            norm_df = df.copy()
            norm_df.iloc[:, 2:] = scaler.fit_transform(norm_df.iloc[:, 2:]) ** (2 if scale == 'minmax_square' else 1)
            scaled_data_list.append(norm_df)
            scaler_dict[f'scaler_{i}'] = scaler
            scaler = scaler_map.get(scale)

    elif structure == 'merge':
        norm_df = data_list[0].copy()
        norm_df.iloc[:, 2:] = scaler.fit_transform(norm_df.iloc[:, 2:]) ** (2 if scale == 'minmax_square' else 1)
        scaled_data_list = [norm_df]
        scaler_dict['scaler_merged'] = scaler
    
    print(f'{scale} Normalization done')

    return scaled_data_list, scaler_dict