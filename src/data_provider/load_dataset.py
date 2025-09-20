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
                numerical_feature: list,
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
    
    # 삭제할 컬럼 제거
    if del_feature is not None:
        df_raw = df_raw.drop(columns=del_feature, errors='ignore')
        
    # 2. 결측치 처리
    df_raw = handle_missing_values(df_raw, ord_categorical_feature, 
                                    categorical_feature, numerical_feature)
    
    # 3. Feature 인코딩
    encoded_df, feature_names, encoders = encode_features(
        df_raw,
        ord_categorical_feature,
        categorical_feature,
        numerical_feature
    )
    
    # 4. 데이터 분할
    if split_date is not None:
        # 날짜 기준 분할
        data_dict = split_by_date(encoded_df, df_raw['데이터 시간'], 
                                    split_date, start_date, end_date)
    else:
        # 비율 기준 분할
        data_dict = split_by_ratio(encoded_df, split_rate)
    
    # 5. 스케일링
    scaler_dict = {}
    if scaler is not None:
        data_dict, scaler = apply_scaling(data_dict, scaler, feature_names)
        scaler_dict['scaler'] = scaler
    
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


def encode_features(df, ord_categorical_feature, categorical_feature, numerical_feature):
    """
    Feature 인코딩 처리
    
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
            feature_names.append(f"{col}_ord")
            ord_encoder_dict[col] = encoder
        
        encoders['ordinal'] = ord_encoder_dict
    
    # 2. Nominal Categorical 인코딩 (One-hot)
    if categorical_feature:
        # Drop first to avoid multicollinearity
        # onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        onehot_encoder = OneHotEncoder(sparse_output=False, drop=None)
        
        valid_cat_features = [col for col in categorical_feature if col in df.columns]
        
        if valid_cat_features:
            onehot_encoded = onehot_encoder.fit_transform(df[valid_cat_features])
            encoded_parts.append(onehot_encoded)
            
            # Feature 이름 생성
            onehot_names = onehot_encoder.get_feature_names_out(valid_cat_features)
            feature_names.extend(onehot_names)
            
            encoders['nominal'] = onehot_encoder
    
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
        '팬속도': ['정지', '저속', '중속', '고속'],
        '댐퍼개도': ['닫힘', '25%', '50%', '75%', '100%'],
        '운전모드': ['정지', '냉방', '난방', '송풍', '제습'],
        '온도설정': ['18', '19', '20', '21', '22', '23', '24', '25', '26'],
        # 추가 매핑 정의...
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


def apply_scaling(data_dict, scaler_method, feature_names):
    """
    학습/검증/테스트 데이터에 동일한 스케일링 적용
    
    Args:
        data_dict (dict): 
            분할된 데이터셋 딕셔너리
            - keys: 'train', 'val', 'test'
            - values: numpy array 형태의 데이터 (n_samples, n_features)
            
        scaler_method (str): 
            스케일링 방법 선택
            - 'standard': StandardScaler (평균 0, 표준편차 1)
            - 'minmax': MinMaxScaler (기본 0~1 범위)
            - 'minmax_square': MinMaxScaler (0~1 범위 명시)
            
        feature_names (list): 
            각 feature의 이름 리스트 (디버깅/로깅용)
            예: ['온도_scaled', '습도_scaled', '모드_onehot_1']
    
    Returns:
        tuple: (data_dict, scaler)
            - data_dict (dict): 스케일링된 데이터 딕셔너리
            - scaler: 학습된 스케일러 객체 (역변환시 필요)
    """
    
    if scaler_method == 'standard':
        scaler = StandardScaler()
    elif scaler_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_method == 'minmax_square':
        scaler = MinMaxScaler(feature_range=(0, 1))
    else:
        raise ValueError(f"Unknown scaling method: {scaler_method}")
    
    scaler.fit(data_dict['train'])
    
    for key in data_dict:
        data_dict[key] = scaler.transform(data_dict[key])
    
    return data_dict, scaler


def split_by_date(encoded_df, date_column, split_date, start_date=None, end_date=None):
    """
    시계열 데이터를 날짜 또는 비율 기준으로 train/val/test 분할
    
    Args:
        encoded_df (pd.DataFrame)
        date_column (pd.Series)
        split_date (str or float or tuple): 
            Train/Val/Test 분할 기준
            
        start_date (str, optional): 
            데이터 사용 시작 날짜 (None이면 데이터 처음부터)
            
        end_date (str, optional): 
            데이터 사용 종료 날짜 (None이면 데이터 끝까지)
    
    Returns:
        dict: 분할된 데이터 딕셔너리
            - 'train': 학습 데이터 (numpy array)
            - 'val': 검증 데이터 (numpy array)
            - 'test': 테스트 데이터 (numpy array)
    """
    
    # 시작/종료 날짜 처리
    if start_date is None:
        start_datetime = date_column.min()
    else:
        start_datetime = pd.to_datetime(start_date)
    
    if end_date is None:
        end_datetime = date_column.max()
    else:
        end_datetime = pd.to_datetime(end_date)
    
    # 유효한 데이터 범위 필터링
    valid_mask = (date_column >= start_datetime) & (date_column <= end_datetime)
    valid_dates = date_column[valid_mask]
    
    if len(valid_dates) == 0:
        raise ValueError(f"No data between {start_date} and {end_date}")
    
    # split_date 타입에 따른 처리
    if isinstance(split_date, str):
        # 날짜로 분할
        split_datetime = pd.to_datetime(split_date)
        
        # 유효성 검사
        if start_datetime >= split_datetime:
            raise ValueError(f"start_date ({start_date}) must be before split_date ({split_date})")
        if split_datetime >= end_datetime:
            raise ValueError(f"split_date ({split_date}) must be before end_date ({end_date})")
        
        # Train: start_date ~ split_date
        train_mask = (date_column >= start_datetime) & (date_column < split_datetime)
        
        # Val과 Test를 위한 중간 지점 계산 (50:50)
        val_end = split_datetime + (end_datetime - split_datetime) / 2
        
        # Val: split_date ~ 중간지점
        val_mask = (date_column >= split_datetime) & (date_column < val_end)
        
        # Test: 중간지점 ~ end_date
        test_mask = (date_column >= val_end) & (date_column <= end_datetime)
        
        # 분할 정보
        train_start, train_end = start_datetime, split_datetime
        val_start, val_end_time = split_datetime, val_end
        test_start, test_end = val_end, end_datetime
        
    elif isinstance(split_date, (int, float)):
        # 단일 비율로 분할
        if not 0 < split_date < 1:
            raise ValueError(f"split_date ratio must be between 0 and 1, got {split_date}")
        
        train_ratio = split_date
        # 나머지를 val과 test가 균등 분할
        remaining_ratio = 1 - train_ratio
        val_ratio = remaining_ratio / 2
        test_ratio = remaining_ratio / 2
        
        # 시간 기반 분할점 계산
        total_duration = end_datetime - start_datetime
        train_end = start_datetime + total_duration * train_ratio
        val_end_time = start_datetime + total_duration * (train_ratio + val_ratio)
        
        # 마스크 생성
        train_mask = (date_column >= start_datetime) & (date_column < train_end)
        val_mask = (date_column >= train_end) & (date_column < val_end_time)
        test_mask = (date_column >= val_end_time) & (date_column <= end_datetime)
        
        # 분할 정보
        train_start = start_datetime
        val_start = train_end
        test_start = val_end_time
        test_end = end_datetime
        
    elif isinstance(split_date, (tuple, list)):
        # (train_ratio, val_ratio) 형태로 분할
        if len(split_date) != 2:
            raise ValueError(f"split_date tuple must have 2 values (train_ratio, val_ratio), got {len(split_date)}")
        
        train_ratio, val_ratio = split_date
        test_ratio = 1 - train_ratio - val_ratio
        
        if test_ratio < 0:
            raise ValueError(f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be <= 1")
        
        if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
            raise ValueError(f"All ratios must be >= 0")
        
        # 시간 기반 분할점 계산
        total_duration = end_datetime - start_datetime
        train_end = start_datetime + total_duration * train_ratio
        val_end_time = start_datetime + total_duration * (train_ratio + val_ratio)
        
        # 마스크 생성
        train_mask = (date_column >= start_datetime) & (date_column < train_end)
        val_mask = (date_column >= train_end) & (date_column < val_end_time)
        test_mask = (date_column >= val_end_time) & (date_column <= end_datetime)
        
        # 분할 정보
        train_start = start_datetime
        val_start = train_end
        test_start = val_end_time
        test_end = end_datetime
        
    else:
        raise ValueError(f"split_date must be str, float, or tuple, got {type(split_date)}")
    
    # 데이터 분할
    data_dict = {
        'train': encoded_df[train_mask].values,
        'val': encoded_df[val_mask].values,
        'test': encoded_df[test_mask].values
    }
    
    # 분할 정보 출력
    print(f"Data split summary:")
    print(f"  Train: {train_start.strftime('%Y-%m-%d %H:%M')} ~ {train_end.strftime('%Y-%m-%d %H:%M')} ({len(data_dict['train'])} samples)")
    print(f"  Val:   {val_start.strftime('%Y-%m-%d %H:%M')} ~ {val_end_time.strftime('%Y-%m-%d %H:%M')} ({len(data_dict['val'])} samples)")
    print(f"  Test:  {test_start.strftime('%Y-%m-%d %H:%M')} ~ {test_end.strftime('%Y-%m-%d %H:%M')} ({len(data_dict['test'])} samples)")
    
    # 비율로 분할한 경우 실제 비율 출력
    if isinstance(split_date, (int, float, tuple, list)):
        total_samples = len(data_dict['train']) + len(data_dict['val']) + len(data_dict['test'])
        if total_samples > 0:
            actual_train_ratio = len(data_dict['train']) / total_samples
            actual_val_ratio = len(data_dict['val']) / total_samples
            actual_test_ratio = len(data_dict['test']) / total_samples
            print(f"  Actual ratios: Train={actual_train_ratio:.1%}, Val={actual_val_ratio:.1%}, Test={actual_test_ratio:.1%}")
    
    if len(data_dict['train']) == 0:
        print("⚠️  Warning: Train set is empty!")
    if len(data_dict['val']) == 0:
        print("⚠️  Warning: Validation set is empty!")
    if len(data_dict['test']) == 0:
        print("⚠️  Warning: Test set is empty!")
    
    return data_dict


def split_by_ratio(encoded_df, split_rate):
    """
    데이터를 지정된 비율로 train/val/test 분할
    
    Args:
        encoded_df (pd.DataFrame)
        split_rate (list of float): [train_ratio, val_ratio, test_ratio] 형태의 분할 비율
    
    Returns:
        dict: 분할된 데이터 딕셔너리
            - 'train': 전체 데이터의 앞부분 (numpy array)
            - 'val': 중간 부분 (numpy array)  
            - 'test': 뒷부분 (numpy array)
    """
    
    n_samples = len(encoded_df)
    train_size = int(n_samples * split_rate[0])
    val_size = int(n_samples * split_rate[1])
    
    data_dict = {
        'train': encoded_df.iloc[:train_size].values,
        'val': encoded_df.iloc[train_size:train_size+val_size].values,
        'test': encoded_df.iloc[train_size+val_size:].values
    }
    
    return data_dict