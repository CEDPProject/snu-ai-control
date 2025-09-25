export CUDA_VISIBLE_DEVICES=0
python main.py \
    --model_name MPCController \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml