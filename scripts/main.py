# Copyright (C) 2023 Elif Cansu YILDIZ
# 
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <http://www.gnu.org/licenses/>.

import typer
import argparse
import yaml
import os
import medai.config as config
from trainer import experiment1, experiment2, experiment3, experiment4, experiment5, experiment6, experiment7

app = typer.Typer()

def get_params_from_config_file(args):
    config_path = os.path.join( config.CONFIG_DIR, args.config)
    with open(config_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)
    return params

############### TRAINING COMMANDS ###############

@app.command()
def train_sasn_vanilla():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment1(params, run_type="train")

@app.command()
def train_sasn_split():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment2(params, run_type="train")

@app.command()
def train_chexnet():
    args = parser.parse_args("--config chexnet_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment3(params, run_type="train")

@app.command()
def train_maskrcnn():
    args = parser.parse_args("--config maskrcnn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment4(params, run_type="train")
    
@app.command()
def train_sasn_vanilla_without_contrastive():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment5(params, run_type="train")

@app.command()
def train_sasn_split_without_contrastive():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment6(params, run_type="train")
    
@app.command()
def train_basic_model():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment7(params, run_type="train")
    
############### TESTING COMMANDS ###############

@app.command()
def test_sasn_vanilla():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment1(params, run_type="test")

@app.command()
def test_sasn_split():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment2(params, run_type="test")

@app.command()
def test_chexnet():
    args = parser.parse_args("--config chexnet_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment3(params, run_type="test")
    
@app.command()
def test_sasn_vanilla_without_contrastive():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment5(params, run_type="test")

@app.command()
def test_sasn_split_without_contrastive():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment6(params, run_type="test")
    
############### TRAIN AND TESTING COMMANDS ###############
    
@app.command()
def train_test_sasn_vanilla():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment1(params, run_type="train_test")

@app.command()
def train_test_sasn_split():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment2(params, run_type="train_test")

@app.command()
def train_test_chexnet():
    args = parser.parse_args("--config chexnet_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment3(params, run_type="train_test")
    
@app.command()
def train_test_sasn_vanilla_without_contrastive():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment5(params, run_type="train_test")

@app.command()
def train_test_sasn_split_without_contrastive():
    args = parser.parse_args("--config sasn_config.yaml".split())
    params = get_params_from_config_file(args)
    experiment6(params, run_type="train_test")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog = 'Disease detection training',
                                     description = 'Trains and evaluates deep learning models',)
    parser.add_argument("--config", "-c", required=True)
    #args = parser.parse_args()
    
    app()