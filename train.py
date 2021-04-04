import numpy as np
import os, time, random
import argparse
import json

import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler

from model.model import InvISPNet
from dataset.FiveK_dataset import FiveKDatasetTrain
from config.config import get_arguments

from utils.JPEG import DiffJPEG

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.system('rm tmp')

DiffJPEG = DiffJPEG(differentiable=True, quality=90).cuda()

parser = get_arguments()
parser.add_argument("--out_path", type=str, default="./exps/", help="Path to save checkpoint. ")
parser.add_argument("--resume", dest='resume', action='store_true',  help="Resume training. ")
parser.add_argument("--loss", type=str, default="L1", choices=["L1", "L2"], help="Choose which loss function to use. ")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--aug", dest='aug', action='store_true', help="Use data augmentation.")
args = parser.parse_args()
print("Parsed arguments: {}".format(args))

os.makedirs(args.out_path, exist_ok=True)
os.makedirs(args.out_path+"%s"%args.task, exist_ok=True)
os.makedirs(args.out_path+"%s/checkpoint"%args.task, exist_ok=True)

with open(args.out_path+"%s/commandline_args.yaml"%args.task , 'w') as f:
    json.dump(args.__dict__, f, indent=2)

def main(args):
    # ======================================define the model======================================
    net = InvISPNet(channel_in=3, channel_out=3, block_num=8)
    net.cuda()
    # load the pretrained weight if there exists one
    if args.resume:
        net.load_state_dict(torch.load(args.out_path+"%s/checkpoint/latest.pth"%args.task))
        print("[INFO] loaded " + args.out_path+"%s/checkpoint/latest.pth"%args.task)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.5)    
    
    print("[INFO] Start data loading and preprocessing")
    RAWDataset = FiveKDatasetTrain(opt=args)        
    dataloader = DataLoader(RAWDataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    print("[INFO] Start to train")
    step = 0
    for epoch in range(0, 300):
        epoch_time = time.time()             
        
        for i_batch, sample_batched in enumerate(dataloader):
            step_time = time.time() 

            input, target_rgb, target_raw = sample_batched['input_raw'].cuda(), sample_batched['target_rgb'].cuda(), \
                                        sample_batched['target_raw'].cuda()
            
            reconstruct_rgb = net(input) 
            reconstruct_rgb = torch.clamp(reconstruct_rgb, 0, 1)
            rgb_loss = F.l1_loss(reconstruct_rgb, target_rgb)
            reconstruct_rgb = DiffJPEG(reconstruct_rgb)
            reconstruct_raw = net(reconstruct_rgb, rev=True)
            raw_loss = F.l1_loss(reconstruct_raw, target_raw)
            
            loss = args.rgb_weight * rgb_loss + raw_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print("task: %s Epoch: %d Step: %d || loss: %.5f raw_loss: %.5f rgb_loss: %.5f || lr: %f time: %f"%(
                args.task, epoch, step, loss.detach().cpu().numpy(), raw_loss.detach().cpu().numpy(), 
                rgb_loss.detach().cpu().numpy(), optimizer.param_groups[0]['lr'], time.time()-step_time
            )) 
            step += 1 
        
        torch.save(net.state_dict(), args.out_path+"%s/checkpoint/latest.pth"%args.task)
        if (epoch+1) % 10 == 0:
            # os.makedirs(args.out_path+"%s/checkpoint/%04d"%(args.task,epoch), exist_ok=True)
            torch.save(net.state_dict(), args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
            print("[INFO] Successfully saved "+args.out_path+"%s/checkpoint/%04d.pth"%(args.task,epoch))
        scheduler.step()   
        
        print("[INFO] Epoch time: ", time.time()-epoch_time, "task: ", args.task)    

if __name__ == '__main__':

    torch.set_num_threads(4)
    main(args)
