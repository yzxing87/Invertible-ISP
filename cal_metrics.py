import cv2
import numpy as np
import math
# from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_ssim 
from scipy.misc import imread
from glob import glob

import argparse

parser = argparse.ArgumentParser(description="evaluation codes")

parser.add_argument("--path", type=str, help="Path to evaluate images.")

args = parser.parse_args()

def psnr(img1, img2):
   mse = np.mean( (img1/255. - img2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_raw(img1, img2):
   mse = np.mean( (img1 - img2) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def my_ssim(img1, img2):
    return compare_ssim(img1, img2, data_range=img1.max() - img1.min(), multichannel=True)


def quan_eval(path, suffix="jpg"):
    # path: /disk2/yazhou/projects/IISP/exps/test_final_unet_globalEDV2/
    # ours
    gt_imgs = sorted(glob(path+"tar*.%s"%suffix))
    pred_imgs = sorted(glob(path+"pred*.%s"%suffix))

    # with open(split_path + "test_gt.txt", 'r') as f_gt, open(split_path+"test_rgb.txt","r") as f_rgb:
    #     gt_imgs = [line.rstrip() for line in f_gt.readlines()]
    #     pred_imgs = [line.rstrip() for line in f_rgb.readlines()]

    assert len(gt_imgs) == len(pred_imgs)

    psnr_avg = 0.
    ssim_avg = 0.
    for i in range(len(gt_imgs)):
        gt = imread(gt_imgs[i])
        pred = imread(pred_imgs[i])
        psnr_temp = psnr(gt, pred)
        psnr_avg += psnr_temp
        ssim_temp = my_ssim(gt, pred)
        ssim_avg += ssim_temp

        print("psnr: ", psnr_temp)
        print("ssim: ", ssim_temp)

    psnr_avg /= float(len(gt_imgs))
    ssim_avg /= float(len(gt_imgs))

    print("psnr_avg: ", psnr_avg)
    print("ssim_avg: ", ssim_avg)

    return psnr_avg, ssim_avg

def mse(gt, pred):
    return np.mean((gt-pred)**2)

def mse_raw(path, suffix="npy"):
    gt_imgs = sorted(glob(path+"raw_tar*.%s"%suffix))
    pred_imgs = sorted(glob(path+"raw_pred*.%s"%suffix))

    # with open(split_path + "test_gt.txt", 'r') as f_gt, open(split_path+"test_rgb.txt","r") as f_rgb:
    #     gt_imgs = [line.rstrip() for line in f_gt.readlines()]
    #     pred_imgs = [line.rstrip() for line in f_rgb.readlines()]
    
    assert len(gt_imgs) == len(pred_imgs)

    mse_avg = 0.
    psnr_avg = 0.
    for i in range(len(gt_imgs)):
        gt = np.load(gt_imgs[i])
        pred = np.load(pred_imgs[i])
        mse_temp = mse(gt, pred)
        mse_avg += mse_temp
        psnr_temp = psnr_raw(gt, pred)
        psnr_avg += psnr_temp

        print("mse: ", mse_temp)
        print("psnr: ", psnr_temp)

    mse_avg /= float(len(gt_imgs))
    psnr_avg /= float(len(gt_imgs))

    print("mse_avg: ", mse_avg)
    print("psnr_avg: ", psnr_avg)

    return mse_avg, psnr_avg

test_full = False

# if test_full:
#     psnr_avg, ssim_avg = quan_eval(ROOT_PATH+"%s/vis_%s_full/"%(args.task, args.ckpt), "jpeg")
#     mse_avg, psnr_avg_raw = mse_raw(ROOT_PATH+"%s/vis_%s_full/"%(args.task, args.ckpt))
# else:
psnr_avg, ssim_avg = quan_eval(args.path, "jpg")
mse_avg, psnr_avg_raw = mse_raw(args.path)    

print("pnsr: {}, ssim: {}, mse: {}, psnr raw: {}".format(psnr_avg, ssim_avg, mse_avg, psnr_avg_raw))


