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
import pickle
import torch
from torchmetrics.classification import BinaryF1Score, BinaryAUROC, BinaryRecall, BinarySpecificity, BinaryAveragePrecision, BinaryPrecision, BinaryAccuracy
import matplotlib.pyplot as plt

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
from medai.utils.eval_utils import get_scores, save_score_table, round_value
import medai.utils.eval_utils as eval_utils
import medai.utils.heatmaps as heatmaps

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
                              config_file:str="baseline_config.yaml",
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
                                                       model_name="baseline",
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

def evaluate_basic_model(model,
                         model_name:str,
                         model_path:str,
                         pickle_path:str,
                         experiment_name:str,
                         threshold:float=0.5,
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
                                                       threshold=threshold)

    print(f"Validation Metric Scores: {val_metric_scores} \nTest Metric Scores: {test_metric_scores}")
    #
    save_score_table(val_metric_scores, experiment_name=experiment_name, threshold=threshold, save_path=val_scores_save_path)
    save_score_table(test_metric_scores, experiment_name=experiment_name, threshold=threshold, save_path=test_scores_save_path)

def show_significance_interval_F1(predictions, targets, threshold, model, return_figure=True):
    predictions = torch.where(predictions > threshold, 1, 0)
    targets = targets.type(torch.int64)
    base_metric = BinaryF1Score()

    output, lower_limit, upper_limit = eval_utils.significance_interval(base_metric, predictions, targets, quantile=0.95, num_bootstraps=100)
    print("Significance interval for F1 Score: [{}, {}]".format(round_value(lower_limit), round_value(upper_limit)))

    if return_figure:
        fig = plt.figure(figsize=(6,6))
        _ = plt.hist(output["raw"].numpy(), 30)
        _ = plt.axvline(x = lower_limit, color = 'red', label = 'axvline - full height')
        _ = plt.axvline(x = upper_limit, color = 'red', label = 'axvline - full height')
        _ = plt.title(model + " model threshold: " + str(round(float(threshold),2)) + " mean: " + str(round(float(output["mean"]), 2)) + " std: " + str(round(float(output["std"]), 2)))
        
        return fig
    
def show_significance_interval_balanced_acc(predictions, targets, threshold, model, return_figure=True):
    predictions = torch.where(predictions > threshold, 1, 0)
    targets = targets.type(torch.int64)
    
    base_metric = BinaryRecall()
    recall_output, recall_lower_limit, recall_upper_limit = eval_utils.significance_interval(base_metric, predictions, targets, quantile=0.95, num_bootstraps=100)
    print("Significance interval for recall: [{}, {}]".format(round_value(recall_lower_limit), round_value(recall_upper_limit)))
    
    base_metric = BinarySpecificity()
    specificity_output, specificity_lower_limit, specificity_upper_limit = eval_utils.significance_interval(base_metric, predictions, targets, quantile=0.95, num_bootstraps=100)
    print("Significance interval for specificity: [{}, {}]".format(round_value(specificity_lower_limit), round_value(specificity_upper_limit)))

    raw_output = (recall_output["raw"] + specificity_output["raw"]) / 2
    lower_limit = (recall_lower_limit + specificity_lower_limit) / 2
    upper_limit = (recall_upper_limit + specificity_upper_limit) / 2
    
    print("Significance interval for balanced accuracy: [{}, {}]".format(round_value(lower_limit), round_value(upper_limit)))
    
    if return_figure:
        fig = plt.figure(figsize=(6,6))
        _ = plt.hist(raw_output.numpy(), 30)
        _ = plt.axvline(x = lower_limit, color = 'red', label = 'axvline - full height')
        _ = plt.axvline(x = upper_limit, color = 'red', label = 'axvline - full height')
        _ = plt.title(model + " model threshold: " + str(round(float(threshold),2)) + " mean: " + str(round(float(raw_output.mean()), 2)) + " std: " + str(round(float(raw_output.std()), 2)))
        
        return fig
    
def show_significance_interval_AUROC(predictions, targets, model, return_figure=True):
    base_metric = BinaryAUROC(thresholds=None)

    output, lower_limit, upper_limit = eval_utils.significance_interval(base_metric, predictions, targets, quantile=0.95, num_bootstraps=100)
    print("Significance interval for AUROC: [{}, {}]".format(round_value(lower_limit), round_value(upper_limit)))

    if return_figure:
        fig = plt.figure(figsize=(6,6))
        _ = plt.hist(output["raw"].numpy(), 30)
        _ = plt.axvline(x = lower_limit, color = 'red', label = 'axvline - full height')
        _ = plt.axvline(x = upper_limit, color = 'red', label = 'axvline - full height')
        _ = plt.title(model + " model mean: " + str(round(float(output["mean"]), 2)) + " std: " + str(round(float(output["std"]), 2)))
        
        return fig

def show_significance_interval_AP(predictions, targets, model, return_figure=True):
    base_metric = BinaryAveragePrecision(thresholds=None)

    output, lower_limit, upper_limit = eval_utils.significance_interval(base_metric, predictions, targets, quantile=0.95, num_bootstraps=100)
    print("Significance interval for average precision: [{}, {}]".format(round_value(lower_limit), round_value(upper_limit)))

    if return_figure:
        fig = plt.figure(figsize=(6,6))
        _ = plt.hist(output["raw"].numpy(), 30)
        _ = plt.axvline(x = lower_limit, color = 'red', label = 'axvline - full height')
        _ = plt.axvline(x = upper_limit, color = 'red', label = 'axvline - full height')
        _ = plt.title(model + " model" + " mean: " + str(round(float(output["mean"]), 2)) + " std: " + str(round(float(output["std"]), 2)))
        
        return fig
    
def get_sasn_heatmaps_bboxes(model_path, model_name="sasn_vanilla", threshold=0.1, to_save=True, save_name="sasn_vanilla"):
    
    config_file = "config.yaml"
    config_file_path = os.path.join(config.CONFIG_DIR, config_file)
    with open(config_file_path, "r") as stream:
        params = yaml.safe_load(stream)

    params["dataloader"]["batch_size"] = 1
    params["dataset"]["train"]["transform"] = None

    sasn_data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    sasn_data_module.setup("test", **params["dataset"]["test"])
    
    if model_name=="sasn_vanilla":
        loaded_model = eval_utils.load_model(model=sasn_vanilla.SiameseNetwork(), path=model_path)
    elif model_name=="sasn_split":
        loaded_model = eval_utils.load_model(model=sasn_split.SiameseNetwork(), path=model_path)

    prob_maps = eval_utils.get_probmaps(loaded_model, sasn_data_module.test_dataloader, model_name=model_name, num_images=553)
    thresholded_probmaps = eval_utils.probmaps_to_masks(prob_maps, threshold=threshold)
    test_bboxes = eval_utils.get_boundingboxes_from_masks(thresholded_probmaps)
    
    if to_save:
        with open("preds_targets/" + save_name + '_heatmaps.pickle', 'wb') as file:
            pickle.dump(prob_maps, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open("preds_targets/" + save_name + '_boxes.pickle', 'wb') as file:
            pickle.dump(test_bboxes, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return prob_maps, test_bboxes

def get_maskrcnn_heatmaps_bboxes(MaskRCNN_model_path, to_save=True, save_name="maskrcnn"):
    config_file = "maskrcnn_config.yaml"
    config_file_path = os.path.join(config.CONFIG_DIR, config_file)
    with open(config_file_path, "r") as stream:
        params = yaml.safe_load(stream)

    params["dataloader"]["batch_size"] = 1

    maskrcnn_data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    maskrcnn_data_module.setup("test", **params["dataset"]["test"])
    
    MaskRCNN_model = eval_utils.load_model(model=MaskRCNN(box_score_thresh=0.5), path=MaskRCNN_model_path)
    
    maskrcnn_test_iter = iter(maskrcnn_data_module.test_dataloader)
    maskrcnn_heatmaps, maskrcnn_boxes = heatmaps.get_MaskRCNN_heatmaps(MaskRCNN_model, maskrcnn_test_iter, num_images=553, print_scores=False)
    maskrcnn_bboxes_nms_applied = eval_utils.apply_nms(maskrcnn_boxes)
    
    if to_save:
        with open("preds_targets/" + save_name + '_masks.pickle', 'wb') as file:
            pickle.dump(maskrcnn_heatmaps, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open("preds_targets/" + save_name + '_boxes.pickle', 'wb') as file:
            boxes = {"without_nms":maskrcnn_boxes, "with_nms":maskrcnn_bboxes_nms_applied}
            pickle.dump(boxes, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return maskrcnn_heatmaps, maskrcnn_boxes, maskrcnn_bboxes_nms_applied

def get_ground_truth_boxes_masks(to_save=True, save_name="ground_truth"):
    config_file = "config.yaml"
    config_file_path = os.path.join(config.CONFIG_DIR, config_file)
    with open(config_file_path, "r") as stream:
        params = yaml.safe_load(stream)

    params["dataloader"]["batch_size"] = 1
    params["dataset"]["train"]["transform"] = None

    sasn_data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    sasn_data_module.setup("test", **params["dataset"]["test"])
    
    target_maps = eval_utils.get_gtmasks(sasn_data_module.test_dataloader, num_images=553)
    thresholded_target_maps = eval_utils.probmaps_to_masks(target_maps, threshold=0.5)
    target_test_bboxes = eval_utils.get_boundingboxes_from_masks(thresholded_target_maps)
    
    if to_save:
        with open("preds_targets/" + save_name + '_masks.pickle', 'wb') as file:
            pickle.dump(target_maps, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open("preds_targets/" + save_name + '_boxes.pickle', 'wb') as file:
            pickle.dump(target_test_bboxes, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    return target_maps, target_test_bboxes

def main():
    evaluate_ChexNet_baseline(model_path=os.path.join(config.BASE_DIR, "logs/DenseNet_model/2023-03-16_00-55-50/version_0/checkpoints/epoch=299-step=3300.ckpt"),
                              pickle_path="scripts/preds_targets/baseline_etherealgorge49_preds_targets.pickle",
                              experiment_name="Baseline (ethereal-gorge-49)",
                              threshold=0.5,
                              config_file="baseline_config.yaml",
                              val_scores_save_path="validation_scores.csv",
                              test_scores_save_path="test_scores.csv")
    
def main2():
    evaluate_SASN(model=sasn_vanilla.SiameseNetwork(),
                  model_name="sasn_vanilla",
                  model_path=os.path.join(config.BASE_DIR, "logs/AASN_model/2023-03-14_16-53-56/version_0/checkpoints/epoch=299-step=51000.ckpt"),
                  pickle_path="scripts/preds_targets/exp2_spicedpie47_preds_vals.pickle",
                  experiment_name="SASN_vanilla (exp2- seed82-spiced-pie-47)",
                  threshold=0.5,
                  config_file="config.yaml",
                  val_scores_save_path="validation_scores.csv",
                  test_scores_save_path="test_scores.csv")

    evaluate_SASN(model=sasn_split.SiameseNetwork(),
                  model_name="sasn_split",
                  model_path=os.path.join(config.BASE_DIR, "logs/AASN_model/2023-03-15_13-28-33/version_0/checkpoints/epoch=299-step=51000.ckpt"),
                  pickle_path="scripts/preds_targets/exp6_stoicyogurt48_preds_vals.pickle",
                  experiment_name="SASN_split (exp6 -stoic-yogurt-48)",
                  threshold=0.5,
                  config_file="config.yaml",
                  val_scores_save_path="validation_scores.csv",
                  test_scores_save_path="test_scores.csv")
 
if __name__ == "__main__":
    main2()