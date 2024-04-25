#!/bin/bash

# 设置环境变量
# export CUDA_VISIBLE_DEVICES=0
# export DATA_DIR=/path/to/data
# export MODEL_DIR=/path/to/model

# 准备数据
python main.py -m fedfast

python main.py -m fedfast  -data m1-1m
# 运行模型训练

python main.py -m fedavg

python main.py -m fedavg  -data m1-1m

python main.py -m gmf

python main.py -m gmf  -data m1-1m
