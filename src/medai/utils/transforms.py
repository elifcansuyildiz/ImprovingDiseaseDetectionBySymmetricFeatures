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
import torchvision
from torchvision import transforms
from PIL import Image

from typing import Dict, Tuple, Union

class Transform:
    
    def __call__(self, x):
        #print("type(x): ", type(x))
        return self.transform(x)
    
    @property
    def config(self) -> Dict:
        raise NotImplementedError
    
class BaseTransform(Transform):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        
    @property
    def config(self) -> Dict:
        return {"name": "base"}
    
class ColorJitterRandomSizedCrop(Transform):
    def __init__(self, resize_dim, image_dim):
        self.resize_dim = resize_dim
        self.image_dim = image_dim
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((resize_dim,resize_dim)),
                transforms.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.RandomResizedCrop(size=(image_dim,image_dim)),
                transforms.ToTensor(),
            ]
        )
    @property
    def config(self) -> Dict:
        return {"name": "ColorJitterRandomSizedCrop","resize dimension": self.resize_dim,"random resize crop dimension": self.image_dim}
    
class Resize(Transform):
    def __init__(self, resize_dim: Tuple):
        self.resize_dim = resize_dim
        self.resize = transforms.Compose(
            [
                #transforms.ToPILImage(),
                transforms.Resize((resize_dim)),
                transforms.ToTensor(),
            ]
        )
        self.transform = self._transform
        
    def _transform(self, x):
        if not isinstance(x, Image.Image):
            to_pil_image = transforms.ToPILImage()
            x = to_pil_image(x)
        x = self.resize(x)
        return x
        
    @property
    def config(self) -> Dict:
        return {"name": "Resize","resize dimension": self.resize_dim}
    
class Padding(Transform):
    def __init__(self, pad_size):
        self.pad_size = pad_size
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Pad(pad_size),
            ]
        )
        
    @property
    def config(self) -> Dict:
        return {"name": "Padding","pad_size": self.pad_size}
    
class CropHorizontalFlip(Transform):
    def __init__(self, img_size, flip_randomness=0.3):
        self.img_size = img_size
        self.flip_randomness = flip_randomness
        self.transform = torch.nn.Sequential(
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=flip_randomness),
        )
    
    @property
    def config(self) -> Dict:
        return {"name": "CropHorizontalFlip","img_size": self.img_size, "flip_randomness": self.flip_randomness}
    
class HorizontalFlip(Transform):
    def __init__(self, flip_randomness=1.0):
        self.transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(p=flip_randomness),
        )
        
    @property
    def config(self) -> Dict:
        return {"name": "HorizontalFlip", "flip_randomness": self.flip_randomness}
    
class RandomHorizontalFlipMulti(torch.nn.Module):
    """
    The purpose is to apply the same randomness of flip transform to the given images
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, imgs):
        if torch.rand(1) > self.p:
            return imgs

        new_imgs = []
        for img in imgs:
            new_imgs.append(torchvision.transforms.functional.hflip(img))
        return new_imgs

class ChexNetAugmentationMultiImages(Transform):
    """
    Applies the transformations in ChexNet paper
    The purpose is to apply the same randomness of fliptransform to both image and target
    Normalization is applied to only the input image not target image.
    """
    def __init__(self, flip_randomness=0.5):
        
        self.random_horzontal_flip = RandomHorizontalFlipMulti(p=flip_randomness)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.transform = self._transform
        
    def _transform(self, imgs):
        imgs = self.random_horzontal_flip(imgs)
        imgs[0] = self.normalize(imgs[0])
        return imgs
    
    @property
    def config(self) -> Dict:
        return {"name": "Augmentation"}

class ChexNetAugmentation(Transform):
    """
    Applies the transformations in ChexNet paper
    """
    def __init__(self, flip_randomness=0.5):

        self.convert = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=flip_randomness),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )
        self.transform = self._transform
        
    def _transform(self, x):
        x = self.convert(x)
        return x
    
    @property
    def config(self) -> Dict:
        return {"name": "Augmentation"}
    
class ImageNetNormalization(Transform):
    """
    Applies the transformations in ChexNet paper
    """
    def __init__(self, flip_randomness=0.5):

        self.convert = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        self.transform = self._transform
        
    def _transform(self, x):
        
        x = self.convert(x)
        return x
    
    @property
    def config(self) -> Dict:
        return {"name": "Augmentation"}
    