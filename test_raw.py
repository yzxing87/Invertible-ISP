import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage
from glob import glob
from tqdm import tqdm

from model.model import InvISPNet
from dataset.FiveK_dataset import FiveKDatasetTest
from config.config import get_arguments

from utils.JPEG import DiffJPEG
from utils.commons import denorm, preprocess_test_patch


os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.system('rm tmp')

DiffJPEG = DiffJPEG(differentiable=True, quality=90).cuda()

parser = get_arguments()
parser.add_argument("--ckpt", type=str, help="Checkpoint path.") 
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint. ")
parser.add_argument("--split_to_patch", dest='split_to_patch', action='store_true', help="Test on patch. ")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))


ckpt_name = args.ckpt.split("/")[-1].split(".")[0]
if args.split_to_patch:
    os.makedirs(args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_metric_%s/"%(args.task, ckpt_name)
else:
    os.makedirs(args.out_path+"%s/results_%s/"%(args.task, ckpt_name), exist_ok=True)
    out_path = args.out_path+"%s/results_%s/"%(args.task, ckpt_name)


def main(args):
    # ======================================define the model============================================
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    device = torch.device("cuda:0")
    
    net.to(device)
    net.eval()
    # load the pretrained weight if there exists one
    if os.path.isfile(args.ckpt):
        net.load_state_dict(torch.load(args.ckpt), strict=False)
        print("[INFO] Loaded checkpoint: {}".format(args.ckpt))
    
    print("[INFO] Start data load and preprocessing") 
    RAWDataset = FiveKDatasetTest(opt=args) 
    dataloader = DataLoader(RAWDataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True) 
    
    input_RGBs = sorted(glob(out_path+"pred*jpg"))
    input_RGBs_names = [path.split("/")[-1].split(".")[0][5:] for path in input_RGBs]    

    print("[INFO] Start test...") 
    for i_batch, sample_batched in enumerate(tqdm(dataloader)):
        step_time = time.time()
        
        input, target_rgb, target_raw = sample_batched['input_raw'].to(device), sample_batched['target_rgb'].to(device), \
                            sample_batched['target_raw'].to(device)
        file_name = sample_batched['file_name'][0]

        if args.split_to_patch:
            input_list, target_rgb_list, target_raw_list = preprocess_test_patch(input, target_rgb, target_raw)
        else:
            # remove [:,:,::2,::2] if you have enough GPU memory to test the full resolution 
            input_list, target_rgb_list, target_raw_list = [input[:,:,::2,::2]], [target_rgb[:,:,::2,::2]], [target_raw[:,:,::2,::2]]
        
        for i_patch in range(len(input_list)):
            file_name_patch = file_name + "_%05d"%i_patch
            idx = input_RGBs_names.index(file_name_patch)
            input_RGB_path = input_RGBs[idx]
            input_RGB = torch.from_numpy(np.array(PILImage.open(input_RGB_path))/255.0).unsqueeze(0).permute(0,3,1,2).float().to(device)
            
            target_raw_patch = target_raw_list[i_patch] 
            
            with torch.no_grad():
                reconstruct_raw = net(input_RGB, rev=True)
            
            pred_raw = reconstruct_raw.detach().permute(0,2,3,1)
            pred_raw = torch.clamp(pred_raw, 0, 1)
            
            target_raw_patch = target_raw_patch.permute(0,2,3,1)
            pred_raw = denorm(pred_raw, 255)
            target_raw_patch = denorm(target_raw_patch, 255)

            pred_raw = pred_raw.cpu().numpy()
            target_raw_patch = target_raw_patch.cpu().numpy().astype(np.float32)

            raw_pred = PILImage.fromarray(np.uint8(pred_raw[0,:,:,0]))
            raw_tar_pred = PILImage.fromarray(np.hstack((np.uint8(target_raw_patch[0,:,:,0]), np.uint8(pred_raw[0,:,:,0]))))
            
            raw_tar = PILImage.fromarray(np.uint8(target_raw_patch[0,:,:,0]))

            raw_pred.save(out_path+"raw_pred_%s_%05d.jpg"%(file_name, i_patch))
            raw_tar.save(out_path+"raw_tar_%s_%05d.jpg"%(file_name, i_patch))
            raw_tar_pred.save(out_path+"raw_gt_pred_%s_%05d.jpg"%(file_name, i_patch))
            
            np.save(out_path+"raw_pred_%s_%05d.npy"%(file_name, i_patch), pred_raw[0,:,:,:]/255.0)
            np.save(out_path+"raw_tar_%s_%05d.npy"%(file_name, i_patch), target_raw_patch[0,:,:,:]/255.0)

            del reconstruct_raw            


if __name__ == '__main__':

    torch.set_num_threads(4)
    main(args)

