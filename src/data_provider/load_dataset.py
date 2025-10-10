import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import warnings

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import warnings

def load_dataset(
                datadir: str,
                scaler: str,
                ord_categorical_feature: list,
                categorical_feature: list,
                cat_encode_method: str,
                numerical_feature: list,
                target_features: list,
                data_version: str,
                environment_name: str,
                split_rate: list = [0.8, 0.1, 0.1],
                del_feature: list = None,
                split_date: str = None,
                end_date: str = None,
                start_date: str = None
                ):
    
    # 1. 데이터 로드
    base_path = os.path.join(datadir, data_version)
    file_name = f"{environment_name}.csv"
    file_path = os.path.join(base_path, file_name)
    df_raw = pd.read_csv(file_path)
    
    # 시간 컬럼 처리
    df_raw['데이터 시간'] = pd.to_datetime(df_raw['데이터 시간'])
    df_raw = df_raw.sort_values('데이터 시간', ascending=True)
    
    # timestamp 별도 저장
    timestamps = df_raw['데이터 시간'].copy()
    
    # 삭제할 컬럼 제거
    if del_feature is not None:
        df_raw = df_raw.drop(columns=del_feature, errors='ignore')
    
    # 이상치 clipping
    clip_thresholds = {
                    '환기유효전력(W)': 1000,
                    '냉난방유효전력(W)': 1000
                }
    if clip_thresholds is not None:
        df_raw = clip_target_outliers(df_raw, target_features, clip_thresholds)
        
    # 2. 결측치 처리
    df_raw = handle_missing_values(df_raw, ord_categorical_feature, 
                                    categorical_feature, numerical_feature)
    
    # 3. Feature 인코딩
    encoded_df, feature_names, encoders = encode_features(
        df_raw,
        ord_categorical_feature,
        categorical_feature,
        numerical_feature,
        cat_encode_method
    )
    
    # 4. 데이터 분할 (timestamp도 함께 분할)
    if split_date is not None:
        # 날짜 기준 분할
        data_dict = split_by_date(encoded_df, timestamps, 
                                    split_date, split_rate, start_date, end_date)
    else:
        # 비율 기준 분할
        data_dict = split_by_ratio(encoded_df, timestamps, split_rate)
    
    # 5. 스케일링 (timestamp는 제외)
    scaler_dict = {}
    if scaler is not None:
        data_dict, scaler, target_scaler = apply_scaling(data_dict, scaler, feature_names, target_features)
        scaler_dict['scaler'] = scaler
        scaler_dict['target_scaler'] = target_scaler
    
    # 인코더 저장
    scaler_dict['encoders'] = encoders
    scaler_dict['feature_names'] = feature_names
    scaler_dict['feature_indices'] = {
        'ordinal': [i for i, name in enumerate(feature_names) 
                    if any(ord_feat in name for ord_feat in ord_categorical_feature)],
        'nominal': [i for i, name in enumerate(feature_names) 
                    if any(cat_feat in name for cat_feat in categorical_feature)],
        'numerical': [i for i, name in enumerate(feature_names) 
                    if any(num_feat in name for num_feat in numerical_feature)]
    }
    
    return data_dict, feature_names, scaler_dict

def clip_target_outliers(df_raw, target_features, clip_thresholds=None):
    """
    Target feature의 극단적 이상치를 clipping으로 제거
    
    Args:
        df_raw: 원본 DataFrame
        target_features: target feature 이름 리스트
        clip_thresholds: dict or float
            - dict: {feature_name: threshold} 형태로 각 feature별 threshold 지정
            - float: 모든 feature에 동일한 threshold 적용
            - None: 자동으로 99.9 percentile 사용
    
    Returns:
        df_raw: clipping이 적용된 DataFrame
    """
    
    print("\n" + "=" * 75)
    print("TARGET FEATURE OUTLIER CLIPPING".center(75))
    print("=" * 75)
    
    for target_feat in target_features:
        if target_feat not in df_raw.columns:
            warnings.warn(f"Target feature '{target_feat}' not found in data")
            continue
        
        # 숫자로 변환 (문자열이 있을 수 있음)
        df_raw[target_feat] = pd.to_numeric(df_raw[target_feat], errors='coerce')
        
        original_data = df_raw[target_feat].copy()
        
        # NaN 제거하고 통계 계산
        valid_data = original_data.dropna()
        
        if len(valid_data) == 0:
            warnings.warn(f"Target feature '{target_feat}' has no valid numeric data")
            continue
        
        # Threshold 결정
        if clip_thresholds is None:
            # 자동: 99.9 percentile 사용
            threshold = np.percentile(valid_data, 99.9)
            print(f"\n[{target_feat}] Auto threshold (p99.9): {threshold:,.0f}")
        elif isinstance(clip_thresholds, dict):
            # Dictionary에서 해당 feature의 threshold 가져오기
            threshold = clip_thresholds.get(target_feat)
            if threshold is None:
                threshold = np.percentile(valid_data, 99.9)
                print(f"\n[{target_feat}] No threshold specified, using p99.9: {threshold:,.0f}")
            else:
                print(f"\n[{target_feat}] Manual threshold: {threshold:,.0f}")
        else:
            # 모든 feature에 동일한 threshold
            threshold = clip_thresholds
            print(f"\n[{target_feat}] Manual threshold: {threshold:,.0f}")
        
        # 통계 출력
        print(f"  Before clipping:")
        print(f"    Min: {valid_data.min():,.0f}")
        print(f"    Max: {valid_data.max():,.0f}")
        print(f"    Mean: {valid_data.mean():,.0f}")
        print(f"    Valid samples: {len(valid_data):,} / {len(original_data):,}")
        
        # Clipping 적용
        df_raw[target_feat] = df_raw[target_feat].clip(upper=threshold)
        
        # 변경된 값 개수
        clipped_count = (valid_data > threshold).sum()
        clipped_rate = clipped_count / len(valid_data) * 100 if len(valid_data) > 0 else 0
        
        print(f"  After clipping:")
        print(f"    Max: {df_raw[target_feat].max():,.0f}")
        print(f"    Clipped: {clipped_count:,} values ({clipped_rate:.2f}%)")
        
        if clipped_count > 0:
            outlier_samples = valid_data[valid_data > threshold].values[:5]
            print(f"    Clipped values (sample): {[f'{v:,.0f}' for v in outlier_samples]}")
    
    print("=" * 75 + "\n")
    
    return df_raw

def encode_features(df, ord_categorical_feature, categorical_feature, numerical_feature, cat_encode_method='onehot'):
    """
    Feature 인코딩 처리
    
    Args:
        cat_encode_method (str): categorical feature 인코딩 방법
            - 'onehot': One-hot encoding (default)
            - 'label': Label encoding (0-1 normalized for SSM stability)
    
    Returns:
        encoded_df: 인코딩된 DataFrame
        feature_names: 인코딩 후 feature 이름 리스트
        encoders: 사용된 인코더 객체들
    """
    
    encoded_parts = []
    feature_names = []
    encoders = {}
    
    # 1. Ordinal Categorical 인코딩
    if ord_categorical_feature:
        ord_encoder_dict = {}
        
        for col in ord_categorical_feature:
            if col not in df.columns:
                warnings.warn(f"Ordinal feature '{col}' not found in data")
                continue
                
            # 순서 정의 (예시 - 실제 데이터에 맞게 수정 필요)
            ordinal_mappings = get_ordinal_mappings(col)
            
            if ordinal_mappings:
                encoder = OrdinalEncoder(categories=[ordinal_mappings])
            else:
                encoder = OrdinalEncoder()
            
            encoded_val = encoder.fit_transform(df[[col]])
            
            # 0-1 범위로 정규화 (SSM 안정성)
            if len(encoder.categories_[0]) > 1:
                encoded_val = encoded_val / (len(encoder.categories_[0]) - 1)
            
            encoded_parts.append(encoded_val)
            feature_names.append(f"{col}")
            ord_encoder_dict[col] = encoder
        
        encoders['ordinal'] = ord_encoder_dict
    
    # 2. Nominal Categorical 인코딩
    if categorical_feature:
        valid_cat_features = [col for col in categorical_feature if col in df.columns]
        
        if valid_cat_features:
            if cat_encode_method == 'onehot':
                # One-hot encoding
                onehot_encoder = OneHotEncoder(sparse_output=False, drop=None)
                onehot_encoded = onehot_encoder.fit_transform(df[valid_cat_features])
                encoded_parts.append(onehot_encoded)
                
                # Feature 이름 생성
                onehot_names = onehot_encoder.get_feature_names_out(valid_cat_features)
                feature_names.extend(onehot_names)
                
                encoders['nominal'] = onehot_encoder
                
            elif cat_encode_method == 'label':
                # Label encoding
                label_encoder_dict = {}
                
                for col in valid_cat_features:
                    # OrdinalEncoder 사용 (LabelEncoder보다 일관성 있음)
                    encoder = OrdinalEncoder()
                    encoded_val = encoder.fit_transform(df[[col]])
                    
                    # 0-1 범위로 정규화 (SSM 안정성)
                    if len(encoder.categories_[0]) > 1:
                        encoded_val = encoded_val / (len(encoder.categories_[0]) - 1)
                    
                    encoded_parts.append(encoded_val)
                    feature_names.append(f"{col}")
                    label_encoder_dict[col] = encoder
                
                encoders['nominal'] = label_encoder_dict
                
            else:
                raise ValueError(f"Unknown cat_encode_method: {cat_encode_method}. Use 'onehot' or 'label'")
    
    # 3. Numerical features
    if numerical_feature:
        valid_num_features = [col for col in numerical_feature if col in df.columns]
        
        if valid_num_features:
            numerical_data = df[valid_num_features].values
            encoded_parts.append(numerical_data)
            feature_names.extend(valid_num_features)
    
    # 4. 모든 인코딩된 부분 결합
    if encoded_parts:
        encoded_array = np.hstack(encoded_parts)
        encoded_df = pd.DataFrame(encoded_array, columns=feature_names)
    else:
        raise ValueError("No valid features found for encoding")
    
    return encoded_df, feature_names, encoders


def get_ordinal_mappings(column_name):
    """
    각 ordinal 변수의 순서 정의
    실제 데이터에 맞게 수정 필요
    """
    ordinal_maps = {
        '온도설정': ['18', '19', '20', '21', '22', '23', '24', '25', '26'],
    }
    
    return ordinal_maps.get(column_name, None)


def handle_missing_values(
    df, 
    ord_categorical_feature, 
    categorical_feature, 
    numerical_feature,
    numerical_method='ffill_bfill',
    categorical_method='mode',
    ordinal_method='mode',
    missing_values=['-', 'NA', 'N/A', '']
):
    """
    데이터프레임의 결측치를 변수 타입별로 적절한 방법으로 처리
    
    Args:
        df (pd.DataFrame): 결측치가 포함된 원본 데이터프레임
        ord_categorical_feature (list): 순서가 있는 범주형 변수 컬럼명 리스트
        categorical_feature (list): 순서가 없는 범주형 변수 컬럼명 리스트
        numerical_feature (list): 연속형 숫자 변수 컬럼명 리스트
            
        numerical_method (str or dict): 수치형 변수 결측치 처리 방법
            - 'ffill_bfill': forward fill → backward fill → 0 
            - 'interpolate': 선형 보간
            - 'mean': 평균값으로 대체
            
        categorical_method (str or dict): 범주형 변수 결측치 처리 방법  
            - 'mode': 최빈값으로 대체
            - 'ffill_bfill': forward fill → backward fill
            - 'unknown': 'unknown' 카테고리로 대체
            
        ordinal_method (str or dict): 순서형 범주 변수 결측치 처리 방법
            - 'mode': 최빈값으로 대체
            - 'ffill_bfill': forward fill → backward fill
            - 'median_category': 중간 순서 카테고리로 대체
    
    Returns:
        pd.DataFrame: 결측치가 처리된 데이터프레임
    """
    
    # 결과 DataFrame 복사
    df_result = df.copy()
    
    # "-" 및 기타 결측값 표시를 NaN으로 변환
    df_result = df_result.replace(missing_values, np.nan)
    
    # 1. Numerical Features 처리
    if numerical_feature:
        df_result = _handle_numerical_missing(
            df_result, 
            numerical_feature, 
            numerical_method
        )
    
    # 2. Categorical Features 처리
    if categorical_feature:
        df_result = _handle_categorical_missing(
            df_result,
            categorical_feature,
            categorical_method
        )
    
    # 3. Ordinal Categorical Features 처리
    if ord_categorical_feature:
        df_result = _handle_ordinal_missing(
            df_result,
            ord_categorical_feature,
            ordinal_method
        )
    
    return df_result


def _handle_numerical_missing(df, features, method):
    """
    수치형 변수 결측치 처리
    
    처리 방법:
    - ffill_bfill: 시계열 데이터에 최적 (앞 값 → 뒤 값 → 0)
    - interpolate: 연속적 변화 가정시 최적 (선형 보간)
    - mean: 단순하고 빠른 처리 (전체 평균값)
    """
    
    for col in features:
        if col not in df.columns:
            continue
            
        # 컬럼별 메소드 확인
        col_method = method if isinstance(method, str) else method.get(col, 'ffill_bfill')
        
        if col_method == 'ffill_bfill':
            # 시계열 최적: 이전값 → 다음값 → 0
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
        elif col_method == 'interpolate':
            # 연속성 가정: 선형 보간
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            # 보간 불가능한 경우 (시작/끝 부분) 평균으로 대체
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
            
        elif col_method == 'mean':
            # 단순 평균값 대체
            mean_val = df[col].mean()
            if pd.isna(mean_val):  # 모든 값이 NaN인 경우
                mean_val = 0
            df[col] = df[col].fillna(mean_val)
            
        else:
            warnings.warn(f"Unknown method '{col_method}' for column '{col}'. Using 'ffill_bfill'.")
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return df


def _handle_categorical_missing(df, features, method):
    """
    범주형 변수 결측치 처리
    
    처리 방법:
    - mode: 가장 빈번한 값 (안전한 기본값)
    - ffill_bfill: 시계열 연속성 가정
    - unknown: 새로운 카테고리로 처리
    """
    
    for col in features:
        if col not in df.columns:
            continue
            
        # 컬럼별 메소드 확인
        col_method = method if isinstance(method, str) else method.get(col, 'mode')
        
        if col_method == 'mode':
            # 최빈값으로 대체
            if not df[col].mode().empty:
                mode_val = df[col].mode()[0]
            else:
                mode_val = 'unknown'
            df[col] = df[col].fillna(mode_val)
            
        elif col_method == 'ffill_bfill':
            # 시계열: 이전값 → 다음값 → unknown
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna('unknown')
            
        elif col_method == 'unknown':
            # 명시적으로 unknown 카테고리로 처리
            df[col] = df[col].fillna('unknown')
            
        else:
            warnings.warn(f"Unknown method '{col_method}' for column '{col}'. Using 'mode'.")
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col] = df[col].fillna(mode_val)
    
    return df


def _handle_ordinal_missing(df, features, method):
    """
    순서형 범주 변수 결측치 처리
    
    처리 방법:
    - mode: 가장 빈번한 값 (안전한 기본값)
    - ffill_bfill: 시계열 연속성 가정
    - median_category: 중간 순서 카테고리 (순서 특성 활용)
    """
    
    for col in features:
        if col not in df.columns:
            continue
            
        # 컬럼별 메소드 확인
        col_method = method if isinstance(method, str) else method.get(col, 'mode')
        
        if col_method == 'mode':
            # 최빈값으로 대체
            if not df[col].mode().empty:
                mode_val = df[col].mode()[0]
            else:
                mode_val = 'unknown'
            df[col] = df[col].fillna(mode_val)
            
        elif col_method == 'ffill_bfill':
            # 시계열: 이전값 → 다음값 → unknown
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna('unknown')
            
        elif col_method == 'median_category':
            # 순서형 카테고리의 중간값 사용
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) > 0:
                # 카테고리를 정렬 (알파벳 순서 또는 자연 순서)
                sorted_vals = sorted(unique_vals)
                median_idx = len(sorted_vals) // 2
                median_cat = sorted_vals[median_idx]
                df[col] = df[col].fillna(median_cat)
            else:
                df[col] = df[col].fillna('unknown')
                
        else:
            warnings.warn(f"Unknown method '{col_method}' for column '{col}'. Using 'mode'.")
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
            df[col] = df[col].fillna(mode_val)
    
    return df


def apply_scaling(data_dict, scaler_method, feature_names, target_features):
    """
    학습/검증/테스트 데이터에 동일한 스케일링 적용
    (timestamp는 제외하고 스케일링)
    
    Args:
        data_dict (dict): 
            분할된 데이터셋 딕셔너리
            - keys: 'train', 'val', 'test'
            - values: {'data': numpy array, 'timestamps': numpy array}
            
        scaler_method (str): 
            스케일링 방법 선택
            - 'standard': StandardScaler (평균 0, 표준편차 1)
            - 'minmax': MinMaxScaler (기본 0~1 범위)
            - 'minmax_square': MinMaxScaler (0~1 범위 명시)
            
        feature_names (list): 
            각 feature의 이름 리스트 (디버깅/로깅용)
    
    Returns:
        tuple: (data_dict, scaler)
            - data_dict (dict): 스케일링된 데이터 딕셔너리
            - scaler: 학습된 스케일러 객체 (역변환시 필요)
    """
    
    if scaler_method == 'standard':
        scaler = StandardScaler()
    elif scaler_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {scaler_method}")
    
    # train 데이터로 scaler fit
    scaler.fit(data_dict['train']['data'])
    
    # 데이터 스케일링
    for key in data_dict:
        data_dict[key]['data'] = scaler.transform(data_dict[key]['data'])
    
    # target_scaler는 전체 scaler의 파라미터에서 target 부분만 추출
    target_indices = [feature_names.index(tf) for tf in target_features]
    
    if scaler_method == 'standard':
        target_scaler = StandardScaler()
        target_scaler.mean_ = scaler.mean_[target_indices]
        target_scaler.scale_ = scaler.scale_[target_indices]
    else:  # minmax
        target_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler.min_ = scaler.min_[target_indices]
        target_scaler.scale_ = scaler.scale_[target_indices]
        target_scaler.data_min_ = scaler.data_min_[target_indices]
        target_scaler.data_max_ = scaler.data_max_[target_indices]
        target_scaler.data_range_ = scaler.data_range_[target_indices]
    
    target_scaler.n_features_in_ = len(target_indices)
        
    return data_dict, scaler, target_scaler

def split_by_date(encoded_df, timestamps, split_date, split_rate=[0.8, 0.1, 0.1],
                  start_date=None, end_date=None):
    """
    시계열 데이터를 2단계로 train/val/test 분할 (timestamp 포함)
    """
    
    # 1. 날짜 범위 설정
    start_dt, end_dt = _get_date_range(timestamps, start_date, end_date)
    
    # 2. split_rate 분리
    train_ratio, val_ratio = split_rate[0], split_rate[1]
    
    # 3. split_date로 train+val과 test 분할점 계산
    trainval_end = _calculate_trainval_split_point(split_date, start_dt, end_dt)
    
    # 4. 1단계 분할: (train+val) vs test
    trainval_data, trainval_timestamps, test_data, test_timestamps = _split_trainval_test(
        encoded_df, timestamps, start_dt, trainval_end, end_dt
    )
    
    # 5. 2단계 분할: train vs val
    train_data, train_timestamps, val_data, val_timestamps = _split_train_val(
        trainval_data, trainval_timestamps, train_ratio, val_ratio
    )
    
    # 6. 결과 딕셔너리 생성
    data_dict = {
        'train': {'data': train_data, 'timestamps': train_timestamps},
        'val': {'data': val_data, 'timestamps': val_timestamps},
        'test': {'data': test_data, 'timestamps': test_timestamps}
    }
    
    # 7. 정보 출력
    _print_two_stage_split_summary(
        data_dict, start_dt, trainval_end, end_dt, 
        train_ratio, val_ratio
    )
    
    return data_dict


def _get_date_range(timestamps, start_date, end_date):
    """날짜 범위 설정"""
    start_dt = pd.to_datetime(start_date) if start_date else timestamps.min()
    end_dt = pd.to_datetime(end_date) if end_date else timestamps.max()
    
    valid_mask = (timestamps >= start_dt) & (timestamps <= end_dt)
    if valid_mask.sum() == 0:
        raise ValueError(f"No data between {start_date} and {end_date}")
    
    return start_dt, end_dt


def _calculate_trainval_split_point(split_date, start_dt, end_dt):
    """train+val과 test의 분할점 계산"""
    if isinstance(split_date, str):
        trainval_end = pd.to_datetime(split_date)
        
        if not (start_dt < trainval_end < end_dt):
            raise ValueError(
                f"split_date ({split_date}) must be between "
                f"start_date ({start_dt:%Y-%m-%d}) and end_date ({end_dt:%Y-%m-%d})"
            )
    
    elif isinstance(split_date, (int, float)):
        if not 0 < split_date < 1:
            raise ValueError(f"split_date ratio must be in (0, 1), got {split_date}")
        
        total_duration = end_dt - start_dt
        trainval_end = start_dt + total_duration * split_date
    
    else:
        raise ValueError(f"split_date must be str or float, got {type(split_date).__name__}")
    
    return trainval_end


def _split_trainval_test(encoded_df, timestamps, start_dt, trainval_end, end_dt):
    """1단계 분할: (train+val) vs test"""
    
    # 마스크 생성
    trainval_mask = (timestamps >= start_dt) & (timestamps < trainval_end)
    test_mask = (timestamps >= trainval_end) & (timestamps <= end_dt)
    
    # 데이터 추출
    trainval_data = encoded_df[trainval_mask].values
    trainval_timestamps = timestamps[trainval_mask].values
    test_data = encoded_df[test_mask].values
    test_timestamps = timestamps[test_mask].values
    
    return trainval_data, trainval_timestamps, test_data, test_timestamps


def _split_train_val(trainval_data, trainval_timestamps, train_ratio, val_ratio):
    """2단계 분할: train vs val"""
    
    n_trainval = len(trainval_data)
    train_size = int(n_trainval * train_ratio)
    
    # 분할
    train_data = trainval_data[:train_size]
    train_timestamps = trainval_timestamps[:train_size]
    val_data = trainval_data[train_size:]
    val_timestamps = trainval_timestamps[train_size:]
    
    return train_data, train_timestamps, val_data, val_timestamps


def _print_two_stage_split_summary(data_dict, start_dt, trainval_end, end_dt, 
                                    train_ratio, val_ratio):
    """2단계 분할 정보 출력"""
    print("\n" + "=" * 75)
    print("TWO-STAGE DATA SPLIT SUMMARY".center(75))
    print("=" * 75)
    
    # Stage 1: train+val vs test
    print("\n[STAGE 1] Train+Val vs Test Split")
    print("-" * 75)
    print(f"  Train+Val: {start_dt:%Y-%m-%d %H:%M} ~ {trainval_end:%Y-%m-%d %H:%M}")
    print(f"  Test:      {trainval_end:%Y-%m-%d %H:%M} ~ {end_dt:%Y-%m-%d %H:%M}")
    
    trainval_samples = len(data_dict['train']['data']) + len(data_dict['val']['data'])
    test_samples = len(data_dict['test']['data'])
    stage1_total = trainval_samples + test_samples
    
    if stage1_total > 0:
        print(f"\n  Train+Val: {trainval_samples:,} samples ({trainval_samples/stage1_total:6.2%})")
        print(f"  Test:      {test_samples:,} samples ({test_samples/stage1_total:6.2%})")
    
    # Stage 2: train vs val
    print(f"\n[STAGE 2] Train vs Val Split (from Train+Val data)")
    print("-" * 75)
    print(f"  Split Ratio: Train={train_ratio:.1%}, Val={val_ratio:.1%}")
    
    if trainval_samples > 0:
        actual_train_ratio = len(data_dict['train']['data']) / trainval_samples
        actual_val_ratio = len(data_dict['val']['data']) / trainval_samples
        
        print(f"\n  Train: {len(data_dict['train']['data']):,} samples ({actual_train_ratio:6.2%} of Train+Val)")
        print(f"  Val:   {len(data_dict['val']['data']):,} samples ({actual_val_ratio:6.2%} of Train+Val)")
    
    # Final Summary
    print(f"\n[FINAL] Overall Split")
    print("-" * 75)
    
    total = sum(len(v['data']) for v in data_dict.values())
    if total > 0:
        for name in ['train', 'val', 'test']:
            count = len(data_dict[name]['data'])
            ratio = count / total
            print(f"  {name.capitalize():5s}: {count:,} samples ({ratio:6.2%} of total)")
    
    # 경고
    warnings_list = []
    if trainval_samples == 0:
        warnings_list.append("⚠️  Train+Val set is EMPTY!")
    if len(data_dict['train']['data']) == 0:
        warnings_list.append("⚠️  TRAIN set is EMPTY!")
    if len(data_dict['val']['data']) == 0:
        warnings_list.append("⚠️  VAL set is EMPTY!")
    if len(data_dict['test']['data']) == 0:
        warnings_list.append("⚠️  TEST set is EMPTY!")
    
    if warnings_list:
        print("\n" + "-" * 75)
        print("WARNINGS:")
        for warning in warnings_list:
            print(f"  {warning}")
    
    print("=" * 75 + "\n")


def split_by_ratio(encoded_df, timestamps, split_rate):
    """
    데이터를 지정된 비율로 train/val/test 분할 (timestamp 포함)
    """
    
    train_ratio, val_ratio, test_ratio = split_rate
    
    n_samples = len(encoded_df)
    
    # 분할 인덱스 계산
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # 데이터 분할
    data_dict = {
        'train': {
            'data': encoded_df.iloc[:train_size].values,
            'timestamps': timestamps.iloc[:train_size].values
        },
        'val': {
            'data': encoded_df.iloc[train_size:train_size+val_size].values,
            'timestamps': timestamps.iloc[train_size:train_size+val_size].values
        },
        'test': {
            'data': encoded_df.iloc[train_size+val_size:].values,
            'timestamps': timestamps.iloc[train_size+val_size:].values
        }
    }
    
    # 정보 출력
    print("\n" + "=" * 70)
    print("DATA SPLIT SUMMARY (Index-based)".center(70))
    print("=" * 70)
    
    total = n_samples
    for name, data in data_dict.items():
        ratio = len(data['data']) / total if total > 0 else 0
        print(f"  {name.capitalize():5s}: {len(data['data']):,} samples ({ratio:6.2%})")
    
    if any(len(data['data']) == 0 for data in data_dict.values()):
        print(f"\n  ⚠️  WARNING: Some splits are empty!")
    
    print("=" * 70 + "\n")
    
    return data_dict