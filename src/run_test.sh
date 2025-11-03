#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# 모델 리스트
# models=("DLinear" "iTransformer" "MICN" "PatchTST" "TimesNet" "TimeXer")
# 모델 리스트
models=("TimeMixer")

# 환경 리스트
environments=("livingroom")

# Target feature 리스트
target_features=("환기유효전력(W)")

# 3중 loop: 모든 모델, 환경, target feature 조합 실행
for model in "${models[@]}"; do
    for env in "${environments[@]}"; do
        for target in "${target_features[@]}"; do
            echo "=========================================="
            echo "Running Model: $model, Environment: $env, Target: $target"
            echo "=========================================="
            
            python main.py \
                --model_name "$model" \
                --default_cfg ./configs/default_setting.yaml \
                --model_cfg ./configs/model_setting.yaml \
                --opts DATASET.environment_name="$env" \
                        DATASET.target_features="['$target']" \
                        DATASET.start_date="2025-08-04 00:00" \
            
            # 각 실행 후 구분선
            echo ""
            echo "Completed: $model on $env with target $target"
            echo ""
        done
    done
done



# 모델 리스트
models=("DLinear" "iTransformer" "MICN" "PatchTST" "TimesNet" "TimeXer")

# 환경 리스트
environments=("livingroom")

# Target feature 리스트
target_features=("냉난방유효전력(W)")

# 3중 loop: 모든 모델, 환경, target feature 조합 실행
for model in "${models[@]}"; do
    for env in "${environments[@]}"; do
        for target in "${target_features[@]}"; do
            echo "=========================================="
            echo "Running Model: $model, Environment: $env, Target: $target"
            echo "=========================================="
            
            python main.py \
                --model_name "$model" \
                --default_cfg ./configs/default_setting.yaml \
                --model_cfg ./configs/model_setting.yaml \
                --opts DATASET.environment_name="$env" \
                        DATASET.target_features="['$target']" \
                        DATASET.start_date="2025-08-05 10:59" \
            
            # 각 실행 후 구분선
            echo ""
            echo "Completed: $model on $env with target $target"
            echo ""
        done
    done
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

