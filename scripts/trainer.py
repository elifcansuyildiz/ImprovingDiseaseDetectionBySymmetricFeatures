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
from torch.utils.data import Subset
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
import yaml
from datetime import datetime
import os
import argparse
from medai.data.loader import ChestDataModule
from medai.data.datasets import ChestXDetDataset
import medai.models.SASN_vanilla as sasn_vanilla
import medai.models.SASN_split as sasn_split
import medai.models.chexnet as chexnet
import medai.models.maskrcnn as maskrcnn
import medai.models.densenet as basicmodel
from medai.models.sasn_wout_contrastive import Classifier2, Classifier3
from medai.utils.loss import ContrastiveLoss
from medai.utils.transforms import ChexNetAugmentation, ChexNetAugmentationMultiImages, ImageNetNormalization
from medai.utils.callbacks import SASNModelLoggingCallback, SASNModelWithoutContrastiveLoggingCallback, ChexnetModelLoggingCallback, MaskRCNNLoggingCallback
import medai.config as config
from medai.config import logger
from medai.utils.helper import set_random_seed
from pytorch_lightning import seed_everything

class Experiment:
    def __init__(self, classifier, model, data_module, optimizer, lr, scheduler, max_epoch, logging_save_dir, 
                 logging_interval, checkpoints_dir, accumulate_grad_batches, ckpt_path, callbacks=None, 
                 wandb_project_name="test-project", accelerator="gpu", devices=1, profiler=None, **kwargs):

        self.data_module = data_module
        self.data_module.setup(stage="fit", **kwargs["dataset"]["train"])
        self.data_module.setup(stage="test", **kwargs["dataset"]["test"])
        
        self.batch_size = self.data_module.train_dataloader.batch_size
        self.ckpt_path = os.path.join(config.BASE_DIR, ckpt_path) if ckpt_path is not None else ""
        
        if profiler == "advanced":
            from pytorch_lightning.profiler import AdvancedProfiler
            logger.info("Profiler info will be saved to a file")
            self.profiler = AdvancedProfiler(filename="advanced_profiler_output")
        elif profiler == "pytorch":
            from pytorch_lightning.profiler import PyTorchProfiler
            logger.info("Profiler info will be saved to a file")
            self.profiler = PyTorchProfiler(filename="pytorch_profiler_output")
        else: #e.g. profiler == "simple"
            self.profiler = profiler
        
        self.save_name = kwargs["model_specific_hyperparams"]["model_name"] + "/" + self.get_datetime_str()
        logger.info(f"Experiment Save Path: {logging_save_dir}/{self.save_name}")
        logger.info(f"Checkpoint path: {self.ckpt_path}")
        
        logging_dataset = Subset(data_module.val_dataset, indices=[1,2,3,4,5])
        
        self.classifier = classifier(model=model, 
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     logging_dataset=logging_dataset,
                                     **kwargs["classifier_specific_params"],)
        
        hparams = {"max epoch": max_epoch, 
                   "model": type(model).__name__,
                   "optimizer": type(optimizer).__name__,
                   "lr": lr, 
                   "scheduler": scheduler,
                   "batch_size": self.batch_size*accumulate_grad_batches,
                   "checkpoint_to_resume_training": "".join(dir+"/" for dir in self.ckpt_path.split("/")[-5:])[:-1] if self.ckpt_path is not None else "",
                   "image_resize": kwargs["dataset"]["train"]["img_resize"],
                   **kwargs["model_specific_hyperparams"],
                  }
        
        tensorboard_logger = pl_loggers.TensorBoardLogger(save_dir=logging_save_dir, name=self.save_name)
        tensorboard_logger.log_hyperparams(hparams)
        
        tensorboard_logger.log_graph(self.classifier)
          
        wandb_logger = WandbLogger(project=wandb_project_name)

        #wandb_logger.experiment.config.update(hparams)
        #wandb_logger.log_table(key="arguments", columns=list(hparams.keys()), data=list(hparams.values()))
        wandb_logger.log_hyperparams(hparams)
        
        self.trainer = pl.Trainer(logger=[tensorboard_logger, wandb_logger], 
                                  callbacks=callbacks,
                                  accelerator=accelerator, 
                                  accumulate_grad_batches=accumulate_grad_batches, 
                                  profiler=self.profiler, # Profiling helps you find bottlenecks in your code by capturing analytics such as how long a function takes or how much memory is used.
                                  default_root_dir=checkpoints_dir, # saves checkpoints to 'some/path/' at every epoch end
                                  max_epochs=max_epoch,
                                  log_every_n_steps=logging_interval, # Default=50
                                  enable_progress_bar=True,
                                  devices=devices,
                                  strategy=None,
                                  deterministic=True,
                                  #limit_train_batches=2, # debug
                                  #limit_val_batches=2, # debug
                                  #limit_test_batches=2, # debug
                                )
    
    def train_validate(self):
        """Train the model.
        """
        log_dataset_info(self.data_module, phase="train_val")
        try:
            self.trainer.fit(self.classifier, self.data_module.train_dataloader, self.data_module.val_dataloader, ckpt_path=self.ckpt_path)
        except FileNotFoundError as e:
            logger.error("Checkpoint file is NOT FOUND!")
        except ValueError as e:
            logger.error("Checkpoint file is NOT FOUND!")
    
    def evaluate(self):
        """Evaluate the model
        """
        log_dataset_info(self.data_module, phase="test")
        try:
            self.trainer.test(self.classifier, self.data_module.test_dataloader, verbose=True, ckpt_path=self.ckpt_path)
        except FileNotFoundError as e:
            logger.error("Checkpoint file is NOT FOUND!")
        except ValueError as e:
            logger.error("Checkpoint file is NOT FOUND!")
    
    def train_validate_evaluate(self):
        """Train and evaluate the model.
        """
        try:
            log_dataset_info(self.data_module, phase="train_val")
            self.trainer.fit(self.classifier, self.data_module.train_dataloader, self.data_module.val_dataloader, ckpt_path=self.ckpt_path)
            
            log_dataset_info(self.data_module, phase="test")
            self.trainer.test(self.classifier, self.data_module.test_dataloader, verbose=True)
        except FileNotFoundError as e:
            logger.error("Checkpoint file is NOT FOUND!")
        except ValueError as e:
            logger.error("Checkpoint file is NOT FOUND!") 
    
    def get_datetime_str(self):
        """returns the date and time of the moment this function is called

        Returns:
            str: date and time
        """
        return str( datetime.now().strftime("%Y-%m-%d_%H-%M-%S") )

def log_dataset_info(data_module, phase="train_val"):
    if phase == "train_val":
        dataset = data_module.train_val_dataset
        logger.warning(f"Number of data for training: {len(data_module.train_dataset)} for validation: {len(data_module.val_dataset)}")
    else: # phase == "test"
        dataset = data_module.test_dataset
        logger.warning(f"Number of data for testing: {len(data_module.test_dataset)}")
    
    logger.info(f"Number of annotations per label: {dataset.num_of_all_labels_per_disease}")
    logger.info(f"Number of image labels: {dataset.num_of_unique_image_labels}")
    logger.info(f"Number of binary labels: {dataset.num_binary_labels}")

def experiment1(params:dict, run_type="train"):
    """Conduct an experiment by training and evaluating SASN_vanilla model.

    Args:
        params (Dict): Parameters for training
    """
    
    #if params["dataset"]["train"]["transform"] is not None:
    #    params["dataset"]["train"]["transform"] = ImageNetNormalization()
    
    seed = 82
    set_random_seed(seed)
    seed_everything(seed, workers=True)  # for pytorch lightning
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = sasn_vanilla.Classifier
    model = sasn_vanilla.SiameseNetwork()
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=params["lr"])
    contrastive_loss = ContrastiveLoss(margin=params["margin"])
    
    params["classifier_specific_params"] = {"contrastive_loss": contrastive_loss, 
                                            "lmbda": params["lmbda"],}
    params["model_specific_hyperparams"] = {"contrastive_loss_function": type(contrastive_loss).__name__,
                                            "margin": params["margin"],
                                            "lambda": params["lmbda"],
                                            "activation_function": "LeakyReLU",
                                            "seed": seed,
                                            "model_name": "SASN_vanilla"}
                                            #"transform": "ImageNetNormalization"}

    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=[SASNModelLoggingCallback()],
                            wandb_project_name="Experiments",
                            **params)
    if run_type=="train":
        experiment.train_validate()
    elif run_type=="test":
        experiment.evaluate()
    elif run_type=="train_test":
        experiment.train_validate_evaluate()
    
def experiment2(params:dict, run_type="train"):
    """Conduct an experiment by training and evaluating SASN_split model.

    Args:
        params (Dict): Parameters for training
    """
    seed = 82
    set_random_seed(seed)
    seed_everything(seed, workers=True)  # for pytorch lightning
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = sasn_split.Classifier
    model = sasn_split.SiameseNetwork()
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=params["lr"])
    contrastive_loss = ContrastiveLoss(margin=params["margin"])
    
    params["classifier_specific_params"] = {"contrastive_loss": contrastive_loss, 
                                            "lmbda": params["lmbda"],}
    params["model_specific_hyperparams"] = {"contrastive_loss_function": type(contrastive_loss).__name__,
                                            "margin": params["margin"],
                                            "lambda": params["lmbda"],
                                            "activation_function": "LeakyReLU",
                                            "seed": seed,
                                            "model_name": "SASN_split"}

    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=[SASNModelLoggingCallback()],
                            wandb_project_name="Experiments",
                            **params)
    
    if run_type=="train":
        experiment.train_validate()
    elif run_type=="test":
        experiment.evaluate()
    elif run_type=="train_test":
        experiment.train_validate_evaluate()
    
def experiment3(params:dict, run_type="train"):
    """Conduct an experiment by training and evaluating CheXNet model.

    Args:
        params (Dict): Parameters for training
    """
    seed = 82
    set_random_seed(seed)
    seed_everything(seed, workers=True)  # for pytorch lightning
    
    if params["dataset"]["train"]["transform"] is not None:
        params["dataset"]["train"]["transform"] = ChexNetAugmentationMultiImages()
    if params["dataset"]["test"]["transform"] is not None:
        params["dataset"]["test"]["transform"] = ChexNetAugmentationMultiImages(flip_randomness=0.0)
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = chexnet.Classifier
    model = chexnet.DenseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    
    params["classifier_specific_params"] = {}
    params["model_specific_hyperparams"] = {"seed": seed,
                                            "model_name": "CheXNet"}

    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=[ChexnetModelLoggingCallback()],
                            wandb_project_name="Experiments",
                            **params)
    
    if run_type=="train":
        experiment.train_validate()
    elif run_type=="test":
        experiment.evaluate()
    elif run_type=="train_test":
        experiment.train_validate_evaluate()

def experiment4(params:dict, run_type="train"):
    """Conduct an experiment by training and evaluating Mask R-CNN model.

    Args:
        params (Dict): Parameters for training
    """
    seed = 82
    set_random_seed(seed)
    seed_everything(seed, workers=True)  # for pytorch lightning
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = maskrcnn.Classifier
    model = maskrcnn.MaskRCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    
    params["classifier_specific_params"] = {}
    params["model_specific_hyperparams"] = {"seed": seed,
                                            "model_name": "MaskRCNN"}

    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=None, #[MaskRCNNLoggingCallback()],
                            wandb_project_name="Experiments", 
                            **params)

    if run_type=="train":
        experiment.train_validate()
    elif run_type=="test":
        logger.error("Testing is not available. Please use Evalution.ipynb for testing this model.")
    
def experiment5(params:dict, run_type="train"):
    """Conduct an experiment by training and evaluating SASN_vanilla model without contrastive learning.

    Args:
        params (Dict): Parameters for training
    """
    
    #if params["dataset"]["train"]["transform"] is not None:
    #    params["dataset"]["train"]["transform"] = ImageNetNormalization()
    
    seed = 82
    set_random_seed(seed)
    seed_everything(seed, workers=True)  # for pytorch lightning
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = Classifier2   # Classifier of SASN model without contrastive learning part
    model = sasn_vanilla.SiameseNetwork(include_feature_comparison=False)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=params["lr"])
    
    params["classifier_specific_params"] = {}
    params["model_specific_hyperparams"] = {"seed": seed,
                                            "model_name": "SASN_vanilla_wihout_contrastive"}
                                            #"transform": "ImageNetNormalization"}

    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=[SASNModelWithoutContrastiveLoggingCallback()],
                            wandb_project_name="Experiments",
                            **params)
    
    if run_type=="train":
        experiment.train_validate()
    elif run_type=="test":
        experiment.evaluate()
    elif run_type=="train_test":
        experiment.train_validate_evaluate()
    
def experiment6(params:dict, run_type="train"):
    """Conduct an experiment by training and evaluating SASN_split model without contrastive learning.

    Args:
        params (Dict): Parameters for training
    """
    seed = 82
    set_random_seed(seed)
    seed_everything(seed, workers=True)  # for pytorch lightning
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = Classifier3  # Classifier of SASN_split model without contrastive learning part
    model = sasn_split.SiameseNetwork(include_feature_comparison=False)
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=params["lr"])
    
    params["classifier_specific_params"] = {}
    params["model_specific_hyperparams"] = {"seed": seed,
                                            "model_name": "SASN_split_without_contrastive"}

    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=[SASNModelWithoutContrastiveLoggingCallback()],
                            wandb_project_name="Experiments",
                            **params)
    
    if run_type=="train":
        experiment.train_validate()
    elif run_type=="test":
        experiment.evaluate()
    elif run_type=="train_test":
        experiment.train_validate_evaluate()

def experiment7(params:dict, run_type="train"):
    """Conduct an experiment by training and evaluating basic model.

    Args:
        params (Dict): Parameters for training
    """
    
    #if params["dataset"]["train"]["transform"] is not None:
    #    params["dataset"]["train"]["transform"] = ImageNetNormalization()
    
    seed = 82
    set_random_seed(seed)
    seed_everything(seed, workers=True)  # for pytorch lightning
    
    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    classifier = basicmodel.Classifier
    model = basicmodel.Network()
    optimizer = torch.optim.Adam(model.get_trainable_parameters(), lr=params["lr"])
    contrastive_loss = ContrastiveLoss(margin=params["margin"])
    
    params["classifier_specific_params"] = {}
    params["model_specific_hyperparams"] = {"seed": seed,
                                            "model_name": "basic_model"}
                                            #"transform": "ImageNetNormalization"}

    experiment = Experiment(classifier,
                            model=model, 
                            data_module=data_module, 
                            optimizer=optimizer, 
                            callbacks=[ChexnetModelLoggingCallback()],
                            wandb_project_name="Experiments",
                            **params)
    if run_type=="train":
        experiment.train_validate()
    elif run_type=="test":
        logger.error("Testing is not available. Please use Evalution.ipynb for testing this model.")