from dataset.kittiloader import *
from torch import nn
import numpy as np
import torch.optim as optimizer
import cv2
import torch
import torch.nn.functional as F
import time
import copy
from scipy.ndimage import median_filter

def save_depth(depth_data, path):
    depth_data_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
    depth_data_8bit = depth_data_normalized.astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth_data_8bit, cv2.COLORMAP_INFERNO)
    cv2.imwrite(path, colored_depth)

def get_performance(model, val_loader, device_str, use_gradient_loss):
    device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()

    with torch.no_grad():
        loss_all = []
        for batch, data in enumerate(val_loader):
            # if (batch % 50 == 0 and batch != 0):
            #     print('Val Batch No. {0}'.format(batch))

            #rgb = data['rgb'].to(device)
            depth = data['depth'].to(device)
            gt = data['gt'].to(device)
            #k = data['k'].to(device)

            estimated_depth = model(depth)
            loss = calculate_loss(estimated_depth[0, :, :, :], gt[0, :, :, :], use_gradient_loss)
            loss_all.append(loss.item())

    val_loss = sum(loss_all) / len(loss_all)
    return val_loss

def save_checkpoint(model, epoch, checkpoint_dir, stats, name):
    """Save a checkpoint file to `checkpoint_dir`."""
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "{}.pth.tar".format(name))
    torch.save(state, filename)

def get_optimizer(net, optim_type, lr, weight_decay):
    if optim_type == 'adam':
        return optimizer.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        return optimizer.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optim_type == 'rmsprop':
        return optimizer.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type. Choose 'adam', 'sgd', or 'rmsprop'.")

def calculate_loss_multi_resolution(reconstructed_img, target_img, use_gradient_loss):
   
    loss_all = 0.0
    for img in reconstructed_img:
        img_resized = F.interpolate(img, size=(480, 640), mode='bilinear', align_corners=False)

        loss_all += calculate_loss(img_resized[0, :, :, :], target_img[0, :, :, :], use_gradient_loss)

    return loss_all / len(reconstructed_img)


def get_performance_multi_resolution(model, val_loader, device_str, use_gradient_loss):
    device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_all = []
    for batch, data in enumerate(val_loader):
        if (batch % 50 == 0 and batch != 0):
            print('Val Batch No. {0}'.format(batch))

        rgb = data['rgb'].to(device)
        depth = data['depth'].to(device)
        gt = data['gt'].to(device)
        k = data['k'].to(device)

        estimated_depths, _ = model(rgb, depth, rgb, depth)
        loss= calculate_loss_multi_resolution(estimated_depths, gt, use_gradient_loss)
        loss_all.append(loss.item())

    val_loss = sum(loss_all) / len(loss_all)
    return val_loss

def gradient_x(img):
    if img.dim() == 3:
        img = img.unsqueeze(0)

    sobel_kernel_x = torch.tensor(
        [[[[1, 0, -1],
           [2, 0, -2],
           [1, 0, -1]]]], dtype=img.dtype, device=img.device
    )
    grad_x = F.conv2d(img, sobel_kernel_x, padding=1)
    grad_x = grad_x.squeeze(0)
    return grad_x


def gradient_y(img):
    if img.dim() == 3:
        img = img.unsqueeze(0)

    sobel_kernel_y = torch.tensor(
        [[[[1, 2, 1],
           [0, 0, 0],
           [-1, -2, -1]]]], dtype=img.dtype, device=img.device
    )

    grad_y = F.conv2d(img, sobel_kernel_y, padding=1)

    grad_y = grad_y.squeeze(0)
    return grad_y


def gradient_loss(input_img, predicted_img):
    diff = input_img - predicted_img

    grad_x = gradient_x(diff)
    grad_y = gradient_y(diff)
    
    grad_x_loss = torch.abs(grad_x).mean()
    grad_y_loss = torch.abs(grad_y).mean()
    
    # Sum both losses
    total_loss = grad_x_loss + grad_y_loss
    return total_loss

def calculate_loss(reconstructed_img, target_img, use_gradient_loss):
    mask = (target_img == 0)
    reconstructed_img = reconstructed_img.masked_fill(mask, 0)

    if (use_gradient_loss):
        loss_metric = torch.sqrt(F.mse_loss(reconstructed_img, target_img))      
        #loss_metric = F.l1_loss(reconstructed_img, target_img)
        loss_gradient = gradient_loss(target_img, reconstructed_img)

        return loss_metric * 0.8 + loss_gradient * 0.2
        
    loss = F.mse_loss(reconstructed_img, target_img)
    #rmse_loss = torch.sqrt(loss)
    return loss
