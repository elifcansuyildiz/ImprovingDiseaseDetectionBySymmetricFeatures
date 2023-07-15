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

import numpy as np
import torch
from torch import nn
from torchvision import transforms
import optuna
import types
from torch.nn.functional import affine_grid, grid_sample
import copy
import matplotlib.pyplot as plt

# img.shape = (rows,cols,3)
def draw_center_vertical_line(img):
    img_c = copy.deepcopy(img)
    cols = img.shape[1]
    img_c[:,cols//2,:] = [1.0,0.0,0.0]
    return img_c

class AffineTransform(nn.Module):
    def __init__(self, tx=0.0, ty=0.0, sx=1.0, sy=1.0, theta=0.0):
        super().__init__()
        self.tx = nn.Parameter(torch.tensor(tx))
        self.ty = nn.Parameter(torch.tensor(ty))
        self.sx = nn.Parameter(torch.tensor(sx))
        self.sy = nn.Parameter(torch.tensor(sy))
        self.theta = nn.Parameter(torch.tensor(theta))
        
    def forward(self, x):
        mat = torch.stack([self.sx*torch.cos(self.theta), -self.sy*torch.sin(self.theta), self.tx, 
                          self.sx*torch.sin(self.theta), self.sy*torch.cos(self.theta), self.ty]).view(1,2,3)
        grid = affine_grid(mat, x.size(), align_corners=False)
        return grid_sample(x, grid, align_corners=False)


class ChestAlignment:
    def __init__(self, use_gpu=True, n_trials=300, feature_weight=1.0, pixel_weight=0.1, theta=(-0.35, 0.35), scale=(0.95, 1/0.95), tx=(-0.2, 0.2), ty=(-0.0, 0.0), pixel_outlier_filter=0.98, feature_outlier_filter=0.98):
        self.use_gpu = use_gpu
        self.feature_weight = feature_weight
        self.pixel_weight = pixel_weight
        self.n_trials = n_trials
        if self.n_trials <= 75:
            print("algorithm works better with n_trials>75")
        self.sample_range = {}
        self.sample_range["theta"] = theta
        self.sample_range["scale"] = scale
        self.sample_range["tx"] = tx
        self.sample_range["ty"] = ty
        self.model = self.create_truncated_resnet()
        if self.use_gpu:
            self.model = self.model.to("cuda")
        self.pixel_outlier_filter = pixel_outlier_filter
        self.feature_outlier_filter = feature_outlier_filter
    
    def create_truncated_resnet(self):
        def forward_override(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            #x = self.layer4(x)
            #x = self.avgpool(x)
            return x   
        
        # Override forward method
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.eval()
        model.forward = types.MethodType(forward_override, model)
        return model
    
    def preprocess(self, x):
        transform = transforms.Compose([
                    transforms.GaussianBlur(7),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])  
        return transform(x)
    
    @torch.no_grad()
    def compute_loss(self, img_t, scale, theta, tx, ty):
        af = AffineTransform(sx=float(scale), sy=float(scale), theta=float(theta), tx=float(tx), ty=float(ty))
        transformed_img = af(img_t)
        mask = torch.ones_like(img_t)
        transformed_mask = af(mask)
        flipped_img = transforms.functional.hflip(transformed_img)
        flipped_mask = transforms.functional.hflip(transformed_mask)

        #plt.imshow(transformed_img.squeeze().permute(1,2,0))
        #plt.show()

        if self.use_gpu:
            out1 = self.model(self.preprocess(transformed_img.cuda())).cpu().numpy()
            out2 = self.model(self.preprocess(flipped_img.cuda())).cpu().numpy()
        else:
            out1 = self.model(self.preprocess(transformed_img)).cpu().numpy()
            out2 = self.model(self.preprocess(flipped_img)).cpu().numpy()    

        # Dist mask
        dist_px = self.create_dist_mat(img_t.shape[2], img_t.shape[3])
        dist_feat = self.create_dist_mat(out1.shape[2], out1.shape[3])

        # resized_mask -> features, combined_mask -> pixels
        combined_mask = flipped_mask * transformed_mask
        mask_resizer = transforms.Resize((out1.shape[2], out1.shape[3]))
        resized_mask = mask_resizer(combined_mask)
        resized_mask = torch.mean(resized_mask, dim=1, keepdim=True).numpy()
        
        # DISABLE DISTANCE TO CENTER PRIOR
        #dist_feat = 1.0
        #dist_px = 1.0
        
        feature_err = dist_feat*resized_mask*(out1-out2)**2
        pixel_err = dist_px*combined_mask.numpy()*(transformed_img.numpy()-flipped_img.numpy())**2
        
        # REMOVE OUTLIERS
        feature_err = feature_err.flatten()
        feature_err = np.sort(feature_err)
        feature_err = feature_err[:int(self.feature_outlier_filter*len(feature_err))]
        #plt.plot(feature_err)
        #plt.show()
        
        pixel_err = pixel_err.flatten()
        pixel_err = np.sort(pixel_err)
        pixel_err = pixel_err[:int(self.pixel_outlier_filter*len(pixel_err))]
        #plt.plot(pixel_err)
        #plt.show()
        
        feature_diff = np.mean(feature_err) / (dist_feat*resized_mask).mean()
        pixel_diff = np.mean(pixel_err) / (dist_px*combined_mask.numpy()).mean()

        return self.feature_weight*feature_diff + self.pixel_weight*pixel_diff
    
    def objective(self, trial, img_t):
        theta = trial.suggest_float("theta", *self.sample_range["theta"])
        scale = trial.suggest_float("scale", *self.sample_range["scale"])
        tx = trial.suggest_float("tx", *self.sample_range["tx"])
        ty = trial.suggest_float("ty", *self.sample_range["ty"])
        return self.compute_loss(img_t, scale, theta, tx, ty)
    
    def create_dist_mat(self, rows, cols, dist_target_min=0.5, dist_target_max=1.0):
        ii, jj = np.meshgrid(range(rows),range(cols), indexing="ij")
        dist = np.sqrt((rows/2-ii)**2 + (cols/2-jj)**2)
        dist = 1.0 - (dist / dist.max())
        dist = dist*(dist_target_max-dist_target_min)+dist_target_min
        return dist
    
    def align_image(self, img, return_aligned_image=True):
        """Chest alignment entry point

        Args:
            img (numpy.array): RGB image with shape (h,w,3)
            return_aligned_image (bool, optional): Defaults to True.

        Returns:
            _type_: Alignment result as a dict
        """
        img_t = torch.tensor(img, dtype=torch.float).permute(2,0,1).unsqueeze(0)
        return self.align_tensor(img_t, return_aligned_image)

    def align_tensor(self, img_t, return_aligned_image=True):
        """Chest alignment entry point

        Args:
            img_t (torch.Tensor): Tensor with shape (1,3,h,w)
            return_aligned_image (bool, optional): Defaults to True.

        Returns:
            _type_: Alignment result as a dict
        """
        #study = optuna.create_study(sampler=optuna.samplers.NSGAIISampler())
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(n_startup_trials=self.n_trials//2, multivariate=True)) # multivariate=True
        study.optimize(lambda trial: self.objective(trial, img_t), n_trials=self.n_trials)
        best_params = copy.deepcopy(study.best_params)
        print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
        
        if return_aligned_image:
            with torch.no_grad():
                af = AffineTransform(sx=best_params["scale"], 
                                    sy=best_params["scale"], 
                                    theta=best_params["theta"], 
                                    tx=best_params["tx"], 
                                    ty=best_params["ty"])
                transformed_img = af(img_t)
                best_params["aligned_image"] = transformed_img.squeeze().permute(1,2,0).numpy()

        return best_params
    
    def unfiltered_costmap_image(self, orig_image, scale=1.0, theta=0.0, tx=0.0, ty=0.0):
        orig_image_t = torch.tensor(orig_image, dtype=torch.float).permute(2,0,1).unsqueeze(0)
        return self.unfiltered_costmap_tensor(orig_image_t, scale, theta, tx, ty)

    @torch.no_grad()
    def unfiltered_costmap_tensor(self, orig_image_t, scale=1.0, theta=0.0, tx=0.0, ty=0.0):
        """Transforms the image with given alignment params and computes pixel and feature 
        costmaps without applying noise filtering.

        Args:
            orig_image_t (torch.Tensor): original image
            scale (float, optional): _description_. Defaults to 1.0.
            theta (float, optional): _description_. Defaults to 0.0.
            tx (float, optional): _description_. Defaults to 0.0.
            ty (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        af = AffineTransform(sx=float(scale), sy=float(scale), theta=float(theta), tx=float(tx), ty=float(ty))
        transformed_img = af(orig_image_t)
        flipped_img = transforms.functional.hflip(transformed_img)

        if self.use_gpu:
            out1 = self.model(self.preprocess(transformed_img.cuda())).cpu().numpy()
            out2 = self.model(self.preprocess(flipped_img.cuda())).cpu().numpy()
        else:
            out1 = self.model(self.preprocess(transformed_img)).cpu().numpy()
            out2 = self.model(self.preprocess(flipped_img)).cpu().numpy()

        feature_err = (out1-out2)**2
        pixel_err = (transformed_img.numpy()-flipped_img.numpy())**2

        feature_err = feature_err.mean(axis=1).squeeze()
        pixel_err = pixel_err.mean(axis=1).squeeze()
        return {"pixel_costmap": pixel_err, "feature_costmap": feature_err}

    def filtered_costmap_image(self, orig_image, scale=1.0, theta=0.0, tx=0.0, ty=0.0):
        orig_image_t = torch.tensor(orig_image, dtype=torch.float).permute(2,0,1).unsqueeze(0)
        return self.filtered_costmap_tensor(orig_image_t, scale, theta, tx, ty)

    @torch.no_grad()
    def filtered_costmap_tensor(self, orig_image_t, scale=1.0, theta=0.0, tx=0.0, ty=0.0):
        """Transforms the image with given alignment params and computes pixel and feature 
        costmaps by applying noise filtering and priors (dist to center linearly).

        Args:
            orig_image_t (torch.Tensor): original image
            scale (float, optional): _description_. Defaults to 1.0.
            theta (float, optional): _description_. Defaults to 0.0.
            tx (float, optional): _description_. Defaults to 0.0.
            ty (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """
        af = AffineTransform(sx=float(scale), sy=float(scale), theta=float(theta), tx=float(tx), ty=float(ty))
        transformed_img = af(orig_image_t)
        mask = torch.ones_like(orig_image_t)
        transformed_mask = af(mask)
        flipped_img = transforms.functional.hflip(transformed_img)
        flipped_mask = transforms.functional.hflip(transformed_mask)

        #plt.imshow(transformed_img.squeeze().permute(1,2,0))
        #plt.show()

        if self.use_gpu:
            out1 = self.model(self.preprocess(transformed_img.cuda())).cpu().numpy()
            out2 = self.model(self.preprocess(flipped_img.cuda())).cpu().numpy()
        else:
            out1 = self.model(self.preprocess(transformed_img)).cpu().numpy()
            out2 = self.model(self.preprocess(flipped_img)).cpu().numpy()

        # Dist mask
        dist_px = self.create_dist_mat(orig_image_t.shape[2], orig_image_t.shape[3])
        dist_feat = self.create_dist_mat(out1.shape[2], out1.shape[3])

        # resized_mask -> features, combined_mask -> pixels
        combined_mask = flipped_mask * transformed_mask
        mask_resizer = transforms.Resize((out1.shape[2], out1.shape[3]))
        resized_mask = mask_resizer(combined_mask)
        resized_mask = torch.mean(resized_mask, dim=1, keepdim=True).numpy()
        
        # DISABLE DISTANCE TO CENTER PRIOR
        #dist_feat = 1.0
        #dist_px = 1.0
        
        feature_err = dist_feat*resized_mask*(out1-out2)**2
        pixel_err = dist_px*combined_mask.numpy()*(transformed_img.numpy()-flipped_img.numpy())**2
        
        # REMOVE OUTLIERS
        feature_shape = feature_err.shape
        feature_err = feature_err.flatten()
        feature_err_idx = np.argsort(feature_err)
        feature_err_remove_idx = feature_err_idx[int(self.feature_outlier_filter*len(feature_err)):]
        feature_err[feature_err_remove_idx] = 0.0
        feature_err = feature_err.reshape(feature_shape)
        
        pixel_shape = pixel_err.shape
        pixel_err = pixel_err.flatten()
        pixel_err_idx = np.argsort(pixel_err)
        pixel_err_remove_idx = pixel_err_idx[int(self.pixel_outlier_filter*len(pixel_err)):]
        pixel_err[pixel_err_remove_idx] = 0.0
        pixel_err = pixel_err.reshape(pixel_shape)

        feature_err = feature_err / (dist_feat*resized_mask).mean()
        pixel_err = pixel_err / (dist_px*combined_mask.numpy()).mean()

        feature_err = feature_err.mean(axis=1).squeeze()
        pixel_err = pixel_err.mean(axis=1).squeeze()
        return {"pixel_costmap": pixel_err, "feature_costmap": feature_err}

if __name__ == "__main__":
    from medai.data.loader import ChestDataModule
    from medai.data.datasets import ChestXDetDataset
    import medai.config as config
    from PIL import Image
    import os
    import yaml
    import argparse
    import pickle
    
    config_file = "sasn_config.yaml"
    config_file_path = os.path.join(config.CONFIG_DIR, config_file)
    with open(config_file_path, "r") as stream:
        params = yaml.safe_load(stream)

    params["dataloader"]["batch_size"] = 1
    params["dataset"]["train"]["transform"] = None

    sasn_data_module = ChestDataModule(dataset_class=ChestXDetDataset, **params["dataloader"])
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--curr_worker", "-i", required=True)
    parser.add_argument("--num_workers", "-n", required=True)
    parser.add_argument('--distribute_to_devices', default=False, action="store_true")
    parser.add_argument("--process_train_set", default=False, action="store_true")
    args = parser.parse_args()  #"--index 1".split()
    curr_worker = int(args.curr_worker)
    num_workers = int(args.num_workers)
    distribute_to_devices = bool(args.distribute_to_devices)
    process_train_set = bool(args.process_train_set)
    
    if distribute_to_devices:
        cuda_device_idx = curr_worker % torch.cuda.device_count()
        torch.cuda.set_device(cuda_device_idx)
        print("Worker-" + str(curr_worker) + " is using cuda device " + str(cuda_device_idx))
    
    if process_train_set:
        sasn_data_module.setup("fit", **params["dataset"]["train"])
        num_files = len(sasn_data_module.train_val_dataset.image_file_paths)
    else:
        sasn_data_module.setup("test", **params["dataset"]["test"])
        num_files = len(sasn_data_module.test_dataset.image_file_paths)

    image_per_worker = int(np.ceil(num_files / num_workers))
    idx_start = curr_worker*image_per_worker
    idx_end = np.min([idx_start+image_per_worker, num_files])

    if process_train_set:
        output_file_name = "results/alignment_result_" + str(curr_worker) + "_" + str(idx_start) + "_" + str(idx_end) + ".pickle"
        file_paths = sasn_data_module.train_val_dataset.image_file_paths[idx_start:idx_end]
    else:
        output_file_name = "results/test_set_alignment_result_" + str(curr_worker) + "_" + str(idx_start) + "_" + str(idx_end) + ".pickle"
        file_paths = sasn_data_module.test_dataset.image_file_paths[idx_start:idx_end]
    
    print("worker:", curr_worker, "start:", idx_start, "end:", idx_end)
    
    #####################
    #exit(0)
    #####################

    results = {}
    chest_alignment = ChestAlignment()
    for img_path in file_paths:
        with Image.open(img_path) as image:
            img = np.array(image.convert('RGB'))/255
        output = chest_alignment.align_image(img, return_aligned_image=False)
        results[img_path] = output
        
        # save after every alignment
        with open(output_file_name, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    