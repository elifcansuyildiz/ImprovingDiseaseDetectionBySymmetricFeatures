#!/bin/bash

mkdir datasets
cd datasets
mkdir ChestX-Det-Dataset
cd ChestX-Det-Dataset
wget http://resource.deepwise.com/ChestX-Det/train_data.zip
sudo apt-get install unzip
unzip train_data.zip -d .
wget http://resource.deepwise.com/ChestX-Det/test_data.zip
unzip test_data.zip -d .
wget https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/main/ChestX_Det_test.json
wget https://raw.githubusercontent.com/Deepwise-AILab/ChestX-Det-Dataset/main/ChestX_Det_train.json
