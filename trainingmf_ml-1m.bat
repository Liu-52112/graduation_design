@echo off
REM 在ml-1m数据集上训练 mf矩阵分解模型

echo Running model with dataset m1-1m and model mf
python main.py -d cuda -data m1-1m -m mf

echo Running model with dataset af and model mf
python main.py -d cuda -data af -m mf

echo Running model with dataset af and model mf
python main.py -d cuda -data ml-100k -m mf

REM xxxx