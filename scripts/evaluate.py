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

import yaml
import os
import medai.config as config
from medai.utils.transforms import Resize
from medai.data.loader import ChestDataModule
from medai.data.datasets import ChestXDetDataset
from medai.utils.transforms import ChexNetAugmentationMultiImages
import medai.models.chexnet as chexnet
import medai.models.SASN_vanilla as sasn_vanilla
import medai.models.SASN_split as sasn_split
from medai.models.maskrcnn import MaskRCNN
from medai.utils.metrics import BinaryMetric
from medai.utils.eval_utils import get_scores, save_score_table
import medai.utils.eval_utils as eval_utils

def evaluate_SASN(model,
                  model_name:str,
                  model_path:str,
                  pickle_path:str,
                  experiment_name:str,
                  threshold:float=0.5,
                  sasn_contrastive=True,
                  config_file:str="config.yaml",
                  val_scores_save_path:str="csv_outputs/validation_scores.csv",
                  test_scores_save_path:str="csv_outputs/test_scores.csv"):

    config_file_path = os.path.join(config.CONFIG_DIR, config_file)
    with open(config_file_path, "r") as stream:
        params = yaml.safe_load(stream)

    params["dataloader"]["batch_size"] = 1
    params["dataset"]["train"]["transform"] = None

    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    data_module.setup("fit", **params["dataset"]["train"])
    data_module.setup("test", **params["dataset"]["test"])
    
    # The following function either loads already computed preds and target values of the model or
    # computes them if the pickle_path does not exist before calculating and outputting the scores. 
    # So this function includes a few subfunctions.
    val_metric_scores, test_metric_scores = get_scores(model=model, 
                                                       data_module=data_module, 
                                                       model_path=model_path, 
                                                       pickle_path=pickle_path,
                                                       model_name=model_name,
                                                       threshold=threshold,
                                                       include_sasn_contrastive=sasn_contrastive)

    print(f"Validation Metric Scores: {val_metric_scores} \nTest Metric Scores: {test_metric_scores}")
    
    save_score_table(val_metric_scores, experiment_name=experiment_name, threshold=threshold, save_path=val_scores_save_path)
    save_score_table(test_metric_scores, experiment_name=experiment_name, threshold=threshold, save_path=test_scores_save_path)

def evaluate_ChexNet_baseline(model_path:str,
                              pickle_path:str,
                              experiment_name:str,
                              threshold:float=0.5,
                              config_file:str="chexnet_config.yaml",
                              val_scores_save_path:str="csv_outputs/validation_scores.csv",
                              test_scores_save_path:str="csv_outputs/test_scores.csv"):
    
    config_file_path = os.path.join(config.CONFIG_DIR, config_file)
    with open(config_file_path, "r") as stream:
        params = yaml.safe_load(stream)

    params["dataloader"]["batch_size"] = 1
    if params["dataset"]["train"]["transform"] is not None:
        params["dataset"]["train"]["transform"] = ChexNetAugmentationMultiImages(flip_randomness=0.0)
    if params["dataset"]["test"]["transform"] is not None:
        params["dataset"]["test"]["transform"] = ChexNetAugmentationMultiImages(flip_randomness=0.0)

    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    data_module.setup("fit", **params["dataset"]["train"])
    data_module.setup("test", **params["dataset"]["test"])
    
    # The following function either loads already computed preds and target values of the model or
    # computes them if the pickle_path does not exist before calculating and outputting the scores. 
    # So this function includes a few subfunctions.
    val_metric_scores, test_metric_scores = get_scores(model=chexnet.DenseNet(), 
                                                       data_module=data_module, 
                                                       model_path=model_path, 
                                                       pickle_path=pickle_path,
                                                       model_name="chexnet",
                                                       threshold=threshold)

    print(f"Validation Metric Scores: {val_metric_scores} \nTest Metric Scores: {test_metric_scores}")

    save_score_table(val_metric_scores, experiment_name=experiment_name, threshold=threshold, save_path=val_scores_save_path)
    save_score_table(test_metric_scores, experiment_name=experiment_name, threshold=threshold, save_path=test_scores_save_path)
    
def evaluate_maskrcnn(model_path:str,
                      pickle_path:str, #"preds_targets/maskrcnn-genial-planet-58.pickle"
                      experiment_name:str, #"Mask R-CNN (genial-planet-58)" 
                      threshold:float=0.5,
                      config_file:str="maskrcnn_config.yaml",
                      test_scores_save_path:str="csv_outputs/test_scores.csv"):  #"csv_outputs/seed82_test_scores.csv"
    
    config_file_path = os.path.join(config.CONFIG_DIR, config_file)
    with open(config_file_path, "r") as stream:
        params = yaml.safe_load(stream)

    params["dataloader"]["batch_size"] = 1

    data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    data_module.setup("fit", **params["dataset"]["train"])
    data_module.setup("test", **params["dataset"]["test"])  
    
    model = MaskRCNN(box_score_thresh=threshold)
    
    loaded_model = eval_utils.load_model(model, model_path)
    loaded_model.eval()
    
    if not os.path.isfile(pickle_path):
        test_pred_labels, test_gt_labels = eval_utils.get_maskrcnn_pred_gt_labels(loaded_model, data_module.test_dataloader)

        eval_utils.save_experiment_preds_targets(val_pred_labels=None, val_target_labels=None,
                                                test_pred_labels=test_pred_labels,
                                                test_target_labels=test_gt_labels,
                                                pickle_path=pickle_path)

    maskrcnn_preds_targets = eval_utils.load_experiment_preds_targets(file_name=pickle_path)
    test_pred_labels, test_gt_labels = maskrcnn_preds_targets["test_pred_labels"], maskrcnn_preds_targets["test_target_labels"]
    print("len(test_pred_labels): ", len(test_pred_labels))

    test_metric_scores = eval_utils.calculate_metrics(test_pred_labels, test_gt_labels)
    print(test_metric_scores)
    save_score_table(test_metric_scores, experiment_name=experiment_name, threshold=threshold, save_path=test_scores_save_path)

########################################################################################################################################
    
def evaluate_sasn_vanilla_experiment():
    with open("experiment_configs/experiment_set2.yaml", "r") as stream:
        experiments = yaml.safe_load(stream)

    validation_scores_csv_path = "csv_outputs/valset_results/experiment_set2_balanced_valset_scores.csv"
    test_scores_csv_path = "csv_outputs/experiment_set2_balanced_testset_scores.csv" 
    
    evaluate_SASN(model=sasn_vanilla.SiameseNetwork(),
                model_name="sasn_vanilla",
                model_path=experiments["SASN_vanilla"]["model_path"],
                pickle_path=experiments["SASN_vanilla"]["labels_pickle_path"],
                experiment_name=experiments["SASN_vanilla"]["exp_name"],
                threshold=0.5,
                config_file="config.yaml",
                val_scores_save_path=validation_scores_csv_path,
                test_scores_save_path=test_scores_csv_path)

def evaluate_sasn_split_experiment():
    with open("experiment_configs/experiment_set2.yaml", "r") as stream:
        experiments = yaml.safe_load(stream)

    validation_scores_csv_path = "csv_outputs/valset_results/experiment_set2_balanced_valset_scores.csv"
    test_scores_csv_path = "csv_outputs/experiment_set2_balanced_testset_scores.csv" 
    
    evaluate_SASN(model=sasn_split.SiameseNetwork(),
              model_name="sasn_split",
              model_path=experiments["SASN_split"]["model_path"],
              pickle_path=experiments["SASN_split"]["labels_pickle_path"],
              experiment_name=experiments["SASN_split"]["exp_name"],
              threshold=0.5,
              config_file="config.yaml",
              val_scores_save_path=validation_scores_csv_path,
              test_scores_save_path=test_scores_csv_path)
    
def evaluate_chexnet_experiment():
    with open("experiment_configs/experiment_set2.yaml", "r") as stream:
        experiments = yaml.safe_load(stream)

    validation_scores_csv_path = "csv_outputs/valset_results/experiment_set2_balanced_valset_scores.csv"
    test_scores_csv_path = "csv_outputs/experiment_set2_balanced_testset_scores.csv" 
    
    evaluate_ChexNet_baseline(model_path=experiments["CheXNet"]["model_path"],
                          pickle_path=experiments["CheXNet"]["labels_pickle_path"],
                          experiment_name=experiments["CheXNet"]["exp_name"],
                          threshold=0.5,
                          config_file="chexnet_config.yaml",
                          val_scores_save_path=validation_scores_csv_path,
                          test_scores_save_path=test_scores_csv_path)
    
def evaluate_maskrcnn_experiment():
    with open("experiment_configs/experiment_set2.yaml", "r") as stream:
        experiments = yaml.safe_load(stream)

    validation_scores_csv_path = "csv_outputs/valset_results/experiment_set2_balanced_valset_scores.csv"
    test_scores_csv_path = "csv_outputs/experiment_set2_balanced_testset_scores.csv"
    
    evaluate_maskrcnn(model_path=experiments["MaskRCNN"]["model_path"],
                  pickle_path=experiments["MaskRCNN"]["labels_pickle_path"],
                  experiment_name=experiments["MaskRCNN"]["exp_name"],
                  threshold=0.5,
                  config_file="maskrcnn_config.yaml",
                  test_scores_save_path=test_scores_csv_path)

if __name__ == "__main__":
    evaluate_sasn_vanilla_experiment()
    evaluate_sasn_split_experiment()
    evaluate_chexnet_experiment()
    evaluate_maskrcnn_experiment()