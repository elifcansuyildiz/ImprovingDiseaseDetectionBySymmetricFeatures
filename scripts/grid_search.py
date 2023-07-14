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

import torch
import numpy as np
import optuna
from optuna.trial import TrialState
#from optuna.integration import PyTorchLightningPruningCallback
from medai.utils.hparam_tuning_callback import PyTorchLightningPruningCallback
import yaml
import os
import argparse
from medai.data.loader import ChestDataModule
from medai.data.datasets import ChestXDetDataset
import medai.models.SASN_vanilla as sasn_vanilla
import medai.models.SASN_split as sasn_split
from medai.utils.callbacks import SASNModelLoggingCallback
from medai.utils.loss import ContrastiveLoss
import medai.config as config
from trainer import Experiment
import wandb
from medai.utils.helper import set_random_seed
from pytorch_lightning import seed_everything

def objective(trial):
    
    seed = trial.suggest_int('seed', 1, 101)
    set_random_seed(seed)
    seed_everything(seed, workers=True)
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = sasn_vanilla.Classifier
    model = sasn_split.SiameseNetwork()
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=params["lr"])
    contrastive_loss = ContrastiveLoss(margin=params["margin"])
    
    params["classifier_specific_params"] = {"contrastive_loss": contrastive_loss, 
                                            "lmbda": params["lmbda"],}
    params["model_specific_hyperparams"] = {"contrastive_loss_function": type(contrastive_loss).__name__,
                                            "margin": params["margin"],
                                            "lmbda": params["lmbda"],
                                            "activation_function": "LeakyReLU",
                                            "seed": seed,
                                            "model_name": "SASN_vanilla"}
    wandb.finish() # It finishes the existing experiment to start a new wandb experiment
    
    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=[SASNModelLoggingCallback()],
                            wandb_project_name="Hyperparam Tuning", #"Tests" #"test-project"
                            **params)
    
    experiment.train_validate()
    return experiment.trainer.callback_metrics["validation_metric/auroc"].item()

def main():
    search_space = {
                    'seed': [2, 12, 22, 32, 42, 52, 62, 82]
                    }
    study = optuna.create_study(storage="sqlite:///db.sqlite3",
                                study_name="Grid Search Balanced Dataset",
                                direction="maximize",
                                sampler=optuna.samplers.GridSampler(search_space),
                                load_if_exists=False)
    study.optimize(objective, n_trials=8)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog = 'Disease detection training',
                                     description = 'Trains and evaluates deep learning models',)
    parser.add_argument("--config", "-c", required=True)
    #args = parser.parse_args()
    args = parser.parse_args("--config config.yaml".split())
    
    config_path = os.path.join(config.CONFIG_DIR, args.config)

    with open(config_path, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)   
    main()