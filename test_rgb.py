import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np
import os, time, random
import argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PILImage

from model.model import InvISPNet
from dataset.FiveK_dataset import FiveKDatasetTest
from config.config import get_arguments

from utils.JPEG import DiffJPEG
from utils.commons import denorm, preprocess_test_patch
from tqdm import tqdm

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.system('rm tmp')

DiffJPEG = DiffJPEG(differentiable=True, quality=90).cuda()

parser = get_arguments()
parser.add_argument("--ckpt", type=str, help="Checkpoint path.") 
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save results. ")
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
            input_patch = input_list[i_patch]
            target_rgb_patch = target_rgb_list[i_patch]
            target_raw_patch = target_raw_list[i_patch] 
            
            with torch.no_grad():
                reconstruct_rgb = net(input_patch)
                reconstruct_rgb = torch.clamp(reconstruct_rgb, 0, 1)
            
            pred_rgb = reconstruct_rgb.detach().permute(0,2,3,1)
            target_rgb_patch = target_rgb_patch.permute(0,2,3,1)
            
            pred_rgb = denorm(pred_rgb, 255)
            target_rgb_patch = denorm(target_rgb_patch, 255)
            pred_rgb = pred_rgb.cpu().numpy()
            target_rgb_patch = target_rgb_patch.cpu().numpy().astype(np.float32)
            
            # print(type(pred_rgb))
            pred = PILImage.fromarray(np.uint8(pred_rgb[0,:,:,:]))
            tar_pred = PILImage.fromarray(np.hstack((np.uint8(target_rgb_patch[0,:,:,:]), np.uint8(pred_rgb[0,:,:,:]))))
            
            tar = PILImage.fromarray(np.uint8(target_rgb_patch[0,:,:,:]))
            
            pred.save(out_path+"pred_%s_%05d.jpg"%(file_name, i_patch), quality=90, subsampling=1)
            tar.save(out_path+"tar_%s_%05d.jpg"%(file_name, i_patch), quality=90, subsampling=1)
            tar_pred.save(out_path+"gt_pred_%s_%05d.jpg"%(file_name, i_patch), quality=90, subsampling=1)
            
            del reconstruct_rgb

if __name__ == '__main__':
    torch.set_num_threads(4)
    main(args)

