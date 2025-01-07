import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import copy

from torch.utils.data import DataLoader
from dataset.nyuloader import DataLoader_NYU
from dpt.models import DPTDepthCompletion

from utils import (
    get_optimizer,
    get_performance,
    save_depth,
    save_checkpoint,
)

def gradient_loss(pred_depth, gt_depth, mask):
    '''
    A simple gradient-based smoothness or edge-aware loss.
    pred_depth, gt_depth shape: [B, 1, H, W]
    mask shape: [B, 1, H, W]  (1=valid pixel, 0=invalid)
    '''
    # Basic L1 masked loss
    l1_loss = F.l1_loss(pred_depth[mask == 1], gt_depth[mask == 1])

    # Compute image gradients
    pred_dx = torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])
    pred_dy = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :])
    smoothness = (pred_dx.mean() + pred_dy.mean())

    return l1_loss + 0.1 * smoothness  # weigh the smoothness as you see fit


def calculate_loss_4ch(pred_depth, gt_depth, mask, use_gradient_loss=True):
    '''
    Combine a masked L1 (or L2) with an optional gradient-based smoothness term.
    Inputs:
      pred_depth: [B, 1, H, W]
      gt_depth:   [B, 1, H, W]
      mask:       [B, 1, H, W]  (1=valid, 0=invalid)
    '''
    if use_gradient_loss:
        return gradient_loss(pred_depth, gt_depth, mask)
    else:
        # standard masked L1
        return F.l1_loss(pred_depth[mask == 1], gt_depth[mask == 1])
    
def train_model(model, 
                train_loader, 
                val_loader, 
                num_epoch, 
                parameter, 
                patience, 
                device_str, 
                use_gradient_loss=True):

    device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.to(device)

    loss_all, loss_index = [], []
    num_iteration = 0
    best_model, best_val_loss = model, float('inf')
    num_bad_epoch = 0

    optim = get_optimizer(model, parameter["optim_type"], parameter["lr"], parameter["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.1, patience=patience
    )

    print('------------------------ Start Training ------------------------')
    t_start = time.time()
    t_step  = t_start

    for epoch in range(num_epoch):
        model.train()
        epoch_losses = []

        for batch_idx, data in enumerate(train_loader):
            rgb   = data['rgb'].to(device, non_blocking=True)   # [B, 3, H, W]
            depth = data['depth'].to(device, non_blocking=True) # [B, 1, H, W]
            gt    = data['gt'].to(device, non_blocking=True)    # [B, 1, H, W]
            mask  = data['mask'].to(device, non_blocking=True)  # [B, 1, H, W], valid=1 or 0

            # Combine RGB + Depth -> 4 channels
            input_4ch = torch.cat([rgb, depth], dim=1)  # shape [B, 4, H, W]

            optim.zero_grad()
            pred_depth = model(input_4ch)  # shape [B, 1, H, W]

            loss = calculate_loss_4ch(pred_depth, gt, mask, use_gradient_loss)
            loss.backward()
            optim.step()

            num_iteration += 1
            loss_all.append(loss.item())
            epoch_losses.append(loss.item())
            loss_index.append(num_iteration)

            if (batch_idx % max(1, (100 // train_loader.batch_size)) == 0) and batch_idx != 0:
                print('Batch No. {0} / Loss: {1:.4f}'.format(batch_idx, loss.item()))
                t_end = time.time()
                print('Delta time {0:.4f} seconds'.format(t_end - t_step))
                t_step = time.time()

                # Save some quick images for debugging
                save_depth(pred_depth[0, 0].detach().cpu().numpy(), 'tmp/color_output.png')
                save_depth(depth[0, 0].detach().cpu().numpy(),      'tmp/color_sparse.png')
                save_depth(gt[0, 0].detach().cpu().numpy(),         'tmp/color_gt.png')

        # End of epoch
        mean_epoch_loss = np.mean(epoch_losses)
        print('Epoch {}/{} - Train Loss: {:.4f}'.format(epoch+1, num_epoch, mean_epoch_loss))

        # ---- Validation ----
        print('Validation...')
        val_loss = get_performance(model, val_loader, device_str, use_gradient_loss)
        print("Validation loss: {:.4f}".format(val_loss))

        # Best model tracking
        if val_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_val_loss = val_loss
            num_bad_epoch = 0
        else:
            num_bad_epoch += 1

        # Early stopping (optional)
        # if (num_bad_epoch >= (patience+3)):
        #     print("Early stopping triggered.")
        #     break

        # LR scheduler step
        scheduler.step(val_loss)
        print("Current learning rate: {:.9f}".format(scheduler.optimizer.param_groups[0]['lr']))

    t_end = time.time()
    print('Training lasted {0:.2f} minutes'.format((t_end - t_start) / 60))
    print("Best validation loss: {:.4f}".format(best_val_loss))
    print('------------------------ Training Done ------------------------')

    stats = {
        'loss': loss_all,
        'loss_ind': loss_index,
    }

    return best_model, best_val_loss, stats


# --------------- Evaluate Performance (example) --------------- #
def get_performance(model, val_loader, device_str, use_gradient_loss=True):
    """
    Example function for validation. We do the same as train:
    - Predict
    - Calculate masked loss
    - Average over the batch
    """
    device = torch.device(device_str if device_str == 'cuda' and torch.cuda.is_available() else 'cpu')
    model.eval()
    loss_vals = []

    with torch.no_grad():
        for data in val_loader:
            rgb   = data['rgb'].to(device)
            depth = data['depth'].to(device)
            gt    = data['gt'].to(device)
            mask  = data['mask'].to(device)

            input_4ch = torch.cat([rgb, depth], dim=1)
            pred_depth = model(input_4ch)

            loss = calculate_loss_4ch(pred_depth, gt, mask, use_gradient_loss)
            loss_vals.append(loss.item())

    mean_loss = np.mean(loss_vals)
    return mean_loss


# --------------- Main Script --------------- #
if __name__ == "__main__":

    # Hyperparams
    output_name = "Test_DPT4CH"
    num_train_epoch = 40
    lr_list = [1e-3, 1e-4]
    wd_list = [1e-7, 1e-6]
    patience = 2
    device_str = 'cuda'

    best_val_loss = float('inf')
    best_model = None
    best_lr = 0
    best_wd = 0
    final_stats = {}

    for lr in lr_list:
        for wd in wd_list:
            # Create Datasets & DataLoaders (NYU example)
            train_dataset = DataLoader_NYU(
                root_path='/oscar/data/jtompki1/cli277/nyuv2/nyuv2', 
                split="train", 
                apply_mask=True, 
                add_noise=False
            )
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)

            val_dataset = DataLoader_NYU(
                root_path='/oscar/data/jtompki1/cli277/nyuv2/nyuv2', 
                split="val", 
                apply_mask=True, 
                add_noise=False
            )
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

            print('Train size: {}, Val size: {}' + str(len(train_loader), str(len(val_loader))))
            print("Learning Rate: {}, Weight Decay: {}".format(lr, wd))

            # Build model
            model_4ch = DPTDepthCompletion(features=256, non_negative=True)
            model_4ch = nn.DataParallel(model_4ch)

            # Parameter configuration
            param_config = {
                "optim_type": 'adam',
                "lr": lr,
                "weight_decay": wd,
                "store_img_training": True
            }

            new_model, val_loss, stats = train_model(
                model=model_4ch,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epoch=num_train_epoch,
                parameter=param_config,
                patience=patience,
                device_str=device_str,
                use_gradient_loss=True
            )

            # Track best
            if val_loss < best_val_loss:
                best_model = copy.deepcopy(new_model)
                best_val_loss = val_loss
                best_lr = lr
                best_wd = wd
                final_stats = stats

    print('------------------------ Training Done ------------------------')
    print('Best validation loss(ALL): {:.4f}'.format(best_val_loss))
    print('Best learning rate(ALL): {}'.format(best_lr))
    print('Best weight decay(ALL): {}'.format(best_wd))
    print('------------------------ Saving Model ------------------------')
    save_checkpoint(best_model, num_train_epoch, "./checkpoints", final_stats, output_name)
    print("Done.")
