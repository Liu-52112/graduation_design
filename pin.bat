@echo off
REM 在pin数据集上训练 

echo Running model with dataset m1-1m and model gmf
python main.py -d cuda -data pin -m gmf

python main.py -d cuda -data pin -m ncf

python main.py -d cuda -data pin -m fedavg

python main.py -d cuda -data pin -m fcf


REM xxxx