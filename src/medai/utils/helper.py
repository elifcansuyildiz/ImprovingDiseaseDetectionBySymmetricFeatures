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
import os
import random

def set_random_seed(random_seed=None):
    """
    Using random seed for numpy and torch
    """
    if(random_seed is None):
        random_seed = 13
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
class StateDictUtils:
    @staticmethod
    def list_to_str(in_list, sep="."):
        out = ""
        for i,s in enumerate(in_list):
            out += s
            if (i!=len(in_list)-1):
                out+= sep
        return out

    @staticmethod
    def remove_prefix(in_str, prefix):
        if in_str.startswith(prefix):
            return in_str[len(prefix):]
        return in_str

    @staticmethod
    def remove_postfix(in_str, postfix):
        if in_str.endswith(postfix):
            return in_str[:-len(postfix)]
        return in_str

    @staticmethod
    def replace_prefix(in_str, old_prefix, new_prefix=""):
        if in_str.startswith(old_prefix):
            return new_prefix + StateDictUtils.remove_prefix(in_str, old_prefix)

    @staticmethod
    def print(in_state_dict, depth, sep="."):
        tree = {}
        for key,val in in_state_dict.items():
            tmp_key = key + "" # copy

            while tmp_key.count(sep) > depth:
                postfix = "." + tmp_key.split(sep)[-1]
                tmp_key = StateDictUtils.remove_postfix(tmp_key, postfix)
            
            if tmp_key not in tree:
                tree[tmp_key] = 1
            else:
                tree[tmp_key] += 1

        stylish_tree = {}
        for key,val in tree.items():
            out = key + "" # copy
            if len(key) > 0 and val > 1:
                out += "."
            if val > 1:
                out += "*"
            stylish_tree[out] = 1
        for k in stylish_tree.keys():
            print(k)