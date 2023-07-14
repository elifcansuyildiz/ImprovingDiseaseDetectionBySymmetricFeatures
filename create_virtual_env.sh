#!/bin/bash

sudo apt install python3.8-venv
python3.8 -m venv medai-venv
source medai-venv/bin/activate
python3.8 -m pip install pip setuptools wheel
pip install -U pip
python3.8 -m pip install -r requirements.txt
python3.8 -m pip install -e .
