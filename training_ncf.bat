@echo off
REM 在ml-1m数据集上训练 mf矩阵分解模型

echo Running model with dataset m1-1m and model ncf
python main.py -d cuda -data m1-1m -m ncf -t 10

echo Running model with dataset m1-1m and model gmf
python main.py -d cuda -data m1-1m -m gmf -t 10

echo Running model with dataset af and model mlp
python main.py -d cuda -data m1-1m -m mlp -t 10

echo Running model with dataset m1-1m and model ncf
python main.py -d cuda -data m1-1m -m ncf -t 20

echo Running model with dataset m1-1m and model gmf
python main.py -d cuda -data m1-1m -m gmf -t 20

echo Running model with dataset af and model mlp
python main.py -d cuda -data m1-1m -m mlp -t 20


REM xxxx