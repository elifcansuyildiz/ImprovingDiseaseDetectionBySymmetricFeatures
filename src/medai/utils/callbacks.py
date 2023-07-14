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
import wandb
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from medai.utils.metrics import BinaryMetric
from medai.utils.visualizer import visualize_predictions, visualize_maskrcnn_predictions

class SASNModelLoggingCallback(pl.Callback):
    def __init__(self):
        super(SASNModelLoggingCallback, self).__init__()
        # To log model outputs
        self.val_data_idxs = torch.asarray([13,16,26])
        self.test_data_idxs = torch.asarray([0,15,21])
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        batch_idxs_to_vis = self.val_data_idxs // pl_module.batch_size
        img_idxs_to_vis = self.val_data_idxs % pl_module.batch_size
        
        tensorboard = pl_module.loggers[0].experiment
        wandb_logger = pl_module.loggers[1].experiment
        
        if batch_idx in batch_idxs_to_vis:
            idx = torch.argwhere(batch_idx==batch_idxs_to_vis)[0][0]
            img_idx = img_idxs_to_vis[idx]
            #Normalization
            dist_map_tmp = pl_module.logged_images["distance_map"][img_idx].detach().cpu().unsqueeze(dim=0)
            #print("min: {} max: {}".format(torch.min(dist_map_tmp), torch.max(dist_map_tmp)))
            dist_map_tmp = dist_map_tmp - torch.min(dist_map_tmp)
            distance_map_normalized = dist_map_tmp / torch.max(dist_map_tmp)
            
            tensorboard.add_image("validation_outputs/target heatmap_"+str(img_idx), pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("validation_outputs/probability map_"+str(img_idx), pl_module.logged_images["probability_map"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("validation_outputs/distance map_"+str(img_idx), distance_map_normalized, pl_module.global_step)
            
            #wandb_logger.log_image(key="validation outputs", images=[probability_map[img_idx].detach().cpu(), distance_map_normalized], caption=["probability map", "distance map"])
            wandb_logger.log({"validation outputs/target heatmap_"+str(img_idx):[wandb.Image(pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), caption="target map")]})
            wandb_logger.log({"validation outputs/probability map_"+str(img_idx):[wandb.Image(pl_module.logged_images["probability_map"][img_idx].detach().cpu(), caption="probability map")]})
            wandb_logger.log({"validation outputs/distance map_"+str(img_idx):[wandb.Image(distance_map_normalized, caption="distance map")]})    
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        #tensorboard = pl_module.logger.experiment
        #tensorboard = pl_module.loggers[0].experiment
        
        results = {"accuracy": BinaryMetric.accuracy(pl_module.val_pred_labels, pl_module.val_target_labels),
                "F1 score": BinaryMetric.f1_score(pl_module.val_pred_labels, pl_module.val_target_labels),
                "precision": BinaryMetric.precision(pl_module.val_pred_labels, pl_module.val_target_labels),
                "recall": BinaryMetric.recall(pl_module.val_pred_labels, pl_module.val_target_labels),
                "specificity": BinaryMetric.specificity(pl_module.val_pred_labels, pl_module.val_target_labels),
                "auroc": BinaryMetric.auroc(pl_module.val_pred_labels, pl_module.val_target_labels)}
        
        pl_module.log("validation_metric/accuracy", results["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/f1_score", results["F1 score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/precision", results["precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/recall", results["recall"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/specificity", results["specificity"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/auroc", results["auroc"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        wandb_logger = pl_module.loggers[1].experiment
        if type(pl_module.model).__name__ == "SiameseNetwork" and pl_module.current_epoch % 10 == 0:
            print("CURRENT EPOCH: ", pl_module.current_epoch)
            fig = visualize_predictions(pl_module.model, pl_module.logging_dataset, pl_module.process_half_image)
            wandb_logger.log({"validation results":fig})

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:

        batch_idxs_to_vis = self.test_data_idxs // pl_module.batch_size
        img_idxs_to_vis = self.test_data_idxs % pl_module.batch_size
        
        tensorboard = pl_module.logger.experiment
        if batch_idx in batch_idxs_to_vis:
            idx = torch.argwhere(batch_idx==batch_idxs_to_vis)[0][0]
            img_idx = img_idxs_to_vis[idx]
            #Normalization
            dist_map_tmp = pl_module.logged_images["distance_map"][img_idx].detach().cpu().unsqueeze(dim=0)
            #print("min: {} max: {}".format(torch.min(dist_map_tmp), torch.max(dist_map_tmp)))
            dist_map_tmp = dist_map_tmp - torch.min(dist_map_tmp)
            distance_map_normalized = dist_map_tmp / torch.max(dist_map_tmp)
            
            tensorboard.add_image("test_outputs/target heatmap_"+str(idx), pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("test_outputs/probability map_"+str(idx), pl_module.logged_images["probability_map"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("test_outputs/distance map_"+str(idx), distance_map_normalized, pl_module.global_step)     

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        results = {"accuracy": BinaryMetric.accuracy(pl_module.test_pred_labels, pl_module.test_target_labels),
                "F1 score": BinaryMetric.f1_score(pl_module.test_pred_labels, pl_module.test_target_labels),
                "precision": BinaryMetric.precision(pl_module.test_pred_labels, pl_module.test_target_labels),
                "recall": BinaryMetric.recall(pl_module.test_pred_labels, pl_module.test_target_labels),
                "specificity": BinaryMetric.specificity(pl_module.test_pred_labels, pl_module.test_target_labels),
                "auroc": BinaryMetric.auroc(pl_module.test_pred_labels, pl_module.test_target_labels)}
        
        pl_module.log("test_metric/accuracy", results["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/f1_score", results["F1 score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/precision", results["precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/recall", results["recall"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/specificity", results["specificity"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/auroc", results["auroc"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tensorboard = pl_module.logger.experiment
        tensorboard.add_pr_curve("test_metric/pr_curve", pl_module.test_target_labels, pl_module.test_pred_labels, 0)
        
class SASNModelWithoutContrastiveLoggingCallback(pl.Callback):
    def __init__(self):
        super(SASNModelWithoutContrastiveLoggingCallback, self).__init__()
        # To log model outputs
        self.val_data_idxs = torch.asarray([13,16,26])
        self.test_data_idxs = torch.asarray([0,15,21])
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        batch_idxs_to_vis = self.val_data_idxs // pl_module.batch_size
        img_idxs_to_vis = self.val_data_idxs % pl_module.batch_size
        
        tensorboard = pl_module.loggers[0].experiment
        wandb_logger = pl_module.loggers[1].experiment
        
        if batch_idx in batch_idxs_to_vis:
            idx = torch.argwhere(batch_idx==batch_idxs_to_vis)[0][0]
            img_idx = img_idxs_to_vis[idx]
            
            tensorboard.add_image("validation_outputs/target heatmap_"+str(img_idx), pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("validation_outputs/probability map_"+str(img_idx), pl_module.logged_images["probability_map"][img_idx].detach().cpu(), pl_module.global_step)
            
            #wandb_logger.log_image(key="validation outputs", images=[probability_map[img_idx].detach().cpu(), distance_map_normalized], caption=["probability map", "distance map"])
            wandb_logger.log({"validation outputs/target heatmap_"+str(img_idx):[wandb.Image(pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), caption="target map")]})
            wandb_logger.log({"validation outputs/probability map_"+str(img_idx):[wandb.Image(pl_module.logged_images["probability_map"][img_idx].detach().cpu(), caption="probability map")]})
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        #tensorboard = pl_module.logger.experiment
        #tensorboard = pl_module.loggers[0].experiment
        
        results = {"accuracy": BinaryMetric.accuracy(pl_module.val_pred_labels, pl_module.val_target_labels),
                "F1 score": BinaryMetric.f1_score(pl_module.val_pred_labels, pl_module.val_target_labels),
                "precision": BinaryMetric.precision(pl_module.val_pred_labels, pl_module.val_target_labels),
                "recall": BinaryMetric.recall(pl_module.val_pred_labels, pl_module.val_target_labels),
                "specificity": BinaryMetric.specificity(pl_module.val_pred_labels, pl_module.val_target_labels),
                "auroc": BinaryMetric.auroc(pl_module.val_pred_labels, pl_module.val_target_labels)}
        
        pl_module.log("validation_metric/accuracy", results["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/f1_score", results["F1 score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/precision", results["precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/recall", results["recall"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/specificity", results["specificity"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/auroc", results["auroc"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:

        batch_idxs_to_vis = self.test_data_idxs // pl_module.batch_size
        img_idxs_to_vis = self.test_data_idxs % pl_module.batch_size
        
        tensorboard = pl_module.logger.experiment
        if batch_idx in batch_idxs_to_vis:
            idx = torch.argwhere(batch_idx==batch_idxs_to_vis)[0][0]
            img_idx = img_idxs_to_vis[idx]
            
            tensorboard.add_image("test_outputs/target heatmap_"+str(idx), pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("test_outputs/probability map_"+str(idx), pl_module.logged_images["probability_map"][img_idx].detach().cpu(), pl_module.global_step)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        results = {"accuracy": BinaryMetric.accuracy(pl_module.test_pred_labels, pl_module.test_target_labels),
                "F1 score": BinaryMetric.f1_score(pl_module.test_pred_labels, pl_module.test_target_labels),
                "precision": BinaryMetric.precision(pl_module.test_pred_labels, pl_module.test_target_labels),
                "recall": BinaryMetric.recall(pl_module.test_pred_labels, pl_module.test_target_labels),
                "specificity": BinaryMetric.specificity(pl_module.test_pred_labels, pl_module.test_target_labels),
                "auroc": BinaryMetric.auroc(pl_module.test_pred_labels, pl_module.test_target_labels)}
        
        pl_module.log("test_metric/accuracy", results["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/f1_score", results["F1 score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/precision", results["precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/recall", results["recall"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/specificity", results["specificity"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/auroc", results["auroc"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tensorboard = pl_module.logger.experiment
        tensorboard.add_pr_curve("test_metric/pr_curve", pl_module.test_target_labels, pl_module.test_pred_labels, 0)

class ChexnetModelLoggingCallback(pl.Callback):
    def __init__(self):
        super(ChexnetModelLoggingCallback, self).__init__()
        # To log model outputs
        self.val_data_idxs = torch.asarray([13,16,26])
        self.test_data_idxs = torch.asarray([0,15,21])
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""

        batch_idxs_to_vis = self.val_data_idxs // pl_module.batch_size
        img_idxs_to_vis = self.val_data_idxs % pl_module.batch_size
        
        tensorboard = pl_module.loggers[0].experiment
        wandb_logger = pl_module.loggers[1].experiment
        
        if batch_idx in batch_idxs_to_vis:
            idx = torch.argwhere(batch_idx==batch_idxs_to_vis)[0][0]
            img_idx = img_idxs_to_vis[idx]
            
            tensorboard.add_image("validation_outputs/target heatmap_"+str(img_idx), pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("validation_outputs/probability map_"+str(img_idx), pl_module.logged_images["pred_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            
            #wandb_logger.log_image(key="validation outputs", images=[probability_map[img_idx].detach().cpu(), distance_map_normalized], caption=["probability map", "distance map"])
            wandb_logger.log({"validation outputs/target heatmap_"+str(img_idx):[wandb.Image(pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), caption="target map")]})
            wandb_logger.log({"validation outputs/probability map_"+str(img_idx):[wandb.Image(pl_module.logged_images["pred_probmap"][img_idx].detach().cpu(), caption="probability map")]})
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        #tensorboard = pl_module.logger.experiment
        #tensorboard = pl_module.loggers[0].experiment
        
        results = {"accuracy": BinaryMetric.accuracy(pl_module.val_pred_labels, pl_module.val_target_labels),
                "F1 score": BinaryMetric.f1_score(pl_module.val_pred_labels, pl_module.val_target_labels),
                "precision": BinaryMetric.precision(pl_module.val_pred_labels, pl_module.val_target_labels),
                "recall": BinaryMetric.recall(pl_module.val_pred_labels, pl_module.val_target_labels),
                "specificity": BinaryMetric.specificity(pl_module.val_pred_labels, pl_module.val_target_labels),
                "auroc": BinaryMetric.auroc(pl_module.val_pred_labels, pl_module.val_target_labels)}
        
        pl_module.log("validation_metric/accuracy", results["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/f1_score", results["F1 score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/precision", results["precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/recall", results["recall"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/specificity", results["specificity"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("validation_metric/auroc", results["auroc"], on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
    
        batch_idxs_to_vis = self.test_data_idxs // pl_module.batch_size
        img_idxs_to_vis = self.test_data_idxs % pl_module.batch_size
        
        tensorboard = pl_module.loggers[0].experiment
        wandb_logger = pl_module.loggers[1].experiment
        
        if batch_idx in batch_idxs_to_vis:
            idx = torch.argwhere(batch_idx==batch_idxs_to_vis)[0][0]
            img_idx = img_idxs_to_vis[idx]
            
            tensorboard.add_image("test_outputs/target heatmap_"+str(idx), pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            tensorboard.add_image("test_outputs/probability map_"+str(idx), pl_module.logged_images["pred_probmap"][img_idx].detach().cpu(), pl_module.global_step)
            
            wandb_logger.log({"test_outputs/target heatmap_"+str(img_idx):[wandb.Image(pl_module.logged_images["target_probmap"][img_idx].detach().cpu(), caption="target map")]})
            wandb_logger.log({"test_outputs/probability map_"+str(img_idx):[wandb.Image(pl_module.logged_images["pred_probmap"][img_idx].detach().cpu(), caption="probability map")]})

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        results = {"accuracy": BinaryMetric.accuracy(pl_module.test_pred_labels, pl_module.test_target_labels),
                "F1 score": BinaryMetric.f1_score(pl_module.test_pred_labels, pl_module.test_target_labels),
                "precision": BinaryMetric.precision(pl_module.test_pred_labels, pl_module.test_target_labels),
                "recall": BinaryMetric.recall(pl_module.test_pred_labels, pl_module.test_target_labels),
                "specificity": BinaryMetric.specificity(pl_module.test_pred_labels, pl_module.test_target_labels),
                "auroc": BinaryMetric.auroc(pl_module.test_pred_labels, pl_module.test_target_labels)}
        
        pl_module.log("test_metric/accuracy", results["accuracy"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/f1_score", results["F1 score"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/precision", results["precision"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/recall", results["recall"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/specificity", results["specificity"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        pl_module.log("test_metric/auroc", results["auroc"], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        tensorboard = pl_module.logger.experiment
        tensorboard.add_pr_curve("test_metric/pr_curve", pl_module.test_target_labels, pl_module.test_pred_labels, 0)
        
class MaskRCNNLoggingCallback(pl.Callback):
    def __init__(self):
        super(MaskRCNNLoggingCallback, self).__init__()
        
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        wandb_logger = pl_module.loggers[1].experiment
        if type(pl_module.model).__name__ == "MaskRCNN" and pl_module.current_epoch % 5 == 0:
            print("CURRENT EPOCH: ", pl_module.current_epoch)
            fig = visualize_maskrcnn_predictions(pl_module.model, pl_module.logging_dataset)
            wandb_logger.log({"validation results":fig})