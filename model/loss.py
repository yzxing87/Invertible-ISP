import torch.nn.functional as F
import torch


def l1_loss(output, target_rgb, target_raw, weight=1.):
    raw_loss = F.l1_loss(output['reconstruct_raw'], target_raw)
    rgb_loss = F.l1_loss(output['reconstruct_rgb'], target_rgb)
    total_loss = raw_loss + weight * rgb_loss
    return total_loss, raw_loss, rgb_loss

def l2_loss(output, target_rgb, target_raw, weight=1.):
    raw_loss = F.mse_loss(output['reconstruct_raw'], target_raw)
    rgb_loss = F.mse_loss(output['reconstruct_rgb'], target_rgb)
    total_loss = raw_loss + weight * rgb_loss
    return total_loss, raw_loss, rgb_loss