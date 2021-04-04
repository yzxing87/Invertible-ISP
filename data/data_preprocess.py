import rawpy
import numpy as np
import glob, os
import colour_demosaicing
import imageio
import argparse
from PIL import Image as PILImage

parser = argparse.ArgumentParser(description="data preprocess")

parser.add_argument("--camera", type=str, default="NIKON_D700", help="Camera Name")
parser.add_argument("--Bayer_Pattern", type=str, default="RGGB", help="Bayer Pattern of RAW")
parser.add_argument("--JPEG_Quality", type=int, default=90, help="Jpeg Quality of the ground truth.")

args = parser.parse_args()
camera_name = args.camera
Bayer_Pattern = args.Bayer_Pattern
JPEG_Quality = args.JPEG_Quality

dng_path = sorted(glob.glob('./' + camera_name + '/DNG/*.dng'))
rgb_target_path = './'+ camera_name + '/RGB/'
raw_input_path = './' + camera_name + '/RAW/'
if not os.path.isdir(rgb_target_path):
    os.mkdir(rgb_target_path)
if not os.path.isdir(raw_input_path):
    os.mkdir(raw_input_path)
    
def flip(raw_img, flip):
    if flip == 3:
        raw_img = np.rot90(raw_img, k=2)
    elif flip == 5:
        raw_img = np.rot90(raw_img, k=1)
    elif flip == 6:
        raw_img = np.rot90(raw_img, k=3)
    else:
        pass
    return raw_img



for path in all_data:
    print("Start Processing %s" % os.path.basename(path))
    raw = rawpy.imread(path)
    file_name = path.split('/')[-1].split('.')[0]
    im = raw.postprocess(use_camera_wb=True,no_auto_bright=True)
    flip_val = raw.sizes.flip
    cwb = raw.camera_whitebalance
    raw_img = raw.raw_image_visible
    if camera_name == 'Canon EOD 5D':
        raw_img = np.maximum(raw_img - 127.0, 0)
    de_raw = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw_img, Bayer_Pattern)
    de_raw = flip(de_raw, flip_val)
    rgb_img = PILImage.fromarray(im).save(rgb_target_path + file_name + '.jpg', quality = JPEG_Quality, subsampling = 1)
    np.savez(raw_input_path + file_name + '.npz', raw=de_raw, wb=cwb)
    
