from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
import torch

class BaseDataset(Dataset):
    def __init__(self, opt):
        self.crop_size = 512
        self.debug_mode = opt.debug_mode
        self.data_path = opt.data_path # dataset path. e.g., ./data/
        self.camera_name = opt.camera 
        self.gamma = opt.gamma

    def norm_img(self, img, max_value):
        img = img / float(max_value)        
        return img

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels
        im = np.expand_dims(raw, axis=2)
        H, W = raw.shape[0], raw.shape[1]
        # RGBG
        out = np.concatenate((im[0:H:2, 0:W:2, :],
                              im[0:H:2, 1:W:2, :],
                              im[1:H:2, 1:W:2, :],
                              im[1:H:2, 0:W:2, :]), axis=2)
        return out
    
    def np2tensor(self, array):
        return torch.Tensor(array).permute(2,0,1)
    
    def center_crop(self, img, crop_size=None):
        H = img.shape[0]
        W = img.shape[1]

        if crop_size is not None:
            th, tw = crop_size[0], crop_size[1]
        else:
            th, tw = self.crop_size, self.crop_size
        x1_img = int(round((W - tw) / 2.))
        y1_img = int(round((H - th) / 2.))
        if img.ndim == 3:
            input_patch = img[y1_img:y1_img + th, x1_img:x1_img + tw, :]
        else:
            input_patch = img[y1_img:y1_img + th, x1_img:x1_img + tw]

        return input_patch

    def load(self, is_train=True):
        # ./data
        # ./data/NIKON D700/RAW, ./data/NIKON D700/RGB 
        # ./data/Canon EOS 5D/RAW,  ./data/Canon EOS 5D/RGB 
        # ./data/NIKON D700_train.txt, ./data/NIKON D700_test.txt 
        # ./data/NIKON D700_train.txt: a0016, ... 
        input_RAWs_WBs = [] 
        target_RGBs = []        
        
        data_path = self.data_path # ./data/ 
        if is_train:
            txt_path = data_path + self.camera_name + "_train.txt"
        else:
            txt_path = data_path + self.camera_name + "_test.txt"

        with open(txt_path, "r") as f_read:
            # valid_camera_list = [os.path.basename(line.strip()).split('.')[0] for line in f_read.readlines()] 
            valid_camera_list = [line.strip() for line in f_read.readlines()] 
        
        if self.debug_mode:
            valid_camera_list = valid_camera_list[:10]
        
        for i,name in enumerate(valid_camera_list): 
            full_name = data_path + self.camera_name 
            input_RAWs_WBs.append(full_name + "/RAW/" + name + ".npz") 
            target_RGBs.append(full_name + "/RGB/" + name + ".jpg") 
            
        return input_RAWs_WBs, target_RGBs 


    def __len__(self):
        return 0

    def __getitem__(self, idx):

        return None
