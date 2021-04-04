import numpy as np


def denorm(img, max_value):
    img = img * float(max_value)
    return img

def preprocess_test_patch(input_image, target_image, gt_image):
    input_patch_list = []
    target_patch_list = []
    gt_patch_list = []
    H = input_image.shape[2]
    W = input_image.shape[3]
    for i in range(3):
        for j in range(3):
            input_patch = input_image[:,:,int(i * H / 3):int((i+1) * H / 3),int(j * W / 3):int((j+1) * W / 3)]
            target_patch = target_image[:,:,int(i * H / 3):int((i+1) * H / 3),int(j * W / 3):int((j+1) * W / 3)]
            gt_patch = gt_image[:,:,int(i * H / 3):int((i+1) * H / 3),int(j * W / 3):int((j+1) * W / 3)]
            input_patch_list.append(input_patch)
            target_patch_list.append(target_patch)
            gt_patch_list.append(gt_patch)
            
    return input_patch_list, target_patch_list, gt_patch_list
