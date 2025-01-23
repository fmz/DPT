import os
import sys
import time
import copy
import logging
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.transforms import Grayscale

import piqa

# Local project imports
from dataset.nyuloader_v2 import NYUDepthDataset
from dpt.models import DPTDepthModel
from utils import (
    get_optimizer,
    save_depth,
    save_rgb,
    save_checkpoint,
    torch_pick_device,
    get_scheduler
)

from debug import Break

###############################################################################
#                        Logging & Misc Setup                                 #
###############################################################################


def setup_logger(log_level=logging.INFO):
    """
    Configure the root logger format and level.
    
    Args:
        log_level (int): one of logging.DEBUG, logging.INFO, etc.
    
    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers if re-running in same process

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger


###############################################################################
#                           Loss Functions                                    #
###############################################################################

class MonoDepthLoss:
    """
    Monocular Depth Loss that combines:
      - Masked L1
      - Optional SSIM-based term (via piqa)
      - Additional placeholders for advanced terms if desired.
    """
    def __init__(self, device):
        self.ssim = piqa.SSIM(n_channels=1).to(device)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor,
                 rgb: torch.Tensor = None,
                 mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies a combined loss to the predicted and target depth maps.

        Args:
            pred (torch.Tensor): Predicted depth, shape [B,1,H,W] or [B,H,W].
            target (torch.Tensor): Ground truth depth, shape [B,1,H,W].
            rgb (torch.Tensor, optional): Optional RGB image for advanced losses. 
            mask (torch.Tensor, optional): Binary mask indicating valid depth pixels.

        Returns:
            torch.Tensor: A scalar loss value.
        """
        # Ensure [B,1,H,W]
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        # (Optional) Crop edges if needed to avoid boundary artifacts
        pred = pred[:, :, 8:-8, 8:-8]
        target = target[:, :, 8:-8, 8:-8]
        if rgb is not None:
            rgb = rgb[:, :, 8:-8, 8:-8]

        # Masked L1
        if mask is not None:
            valid = mask.bool()
            l1_loss = F.mse_loss(pred[valid], target[valid])
        else:
            diff = torch.abs(pred - target)

            # diff_gt_1 = diff.clone()
            # diff_gt_1[diff < 1.0] = 0
            # diff_sq = diff_gt_1 * diff_gt_1
            diff = torch.pow(diff + 1, 3) + diff + 1 # NOT really L1...
            l1_loss = torch.mean(diff)

        Break.point()

        logging.debug(f'L1 loss: {l1_loss}')
        
        loss = l1_loss

        # # Combine with SSIM
        # # We'll do a basic "normalized" approach if pred & target have non-zero ranges
        # max_p, min_p = pred.max(), pred.min()
        # max_t, min_t = target.max(), target.min()

        # if max_p > min_p and max_t > min_t:
        #     pred_norm = (pred - min_p) / (max_p - min_p)
        #     targ_norm = (target - min_t) / (max_t - min_t)

        #     ssim_loss = 1.0 - self.ssim(pred_norm, targ_norm)

        #     logging.debug(f'SSIM loss: {ssim_loss}')

        #     loss = loss + 0.2 * ssim_loss

        # Perform edge-aware-smoothness loss if rgb is available
        if rgb is not None:
            alpha = 10.0
            # partial derivatives
            depth_dx = torch.abs(pred[:,:,:,1:] - pred[:,:,:,:-1])
            depth_dy = torch.abs(pred[:,:,1:,:] - pred[:,:,:-1,:])

            # Convert to grayscale for gradient computation
            to_gs = Grayscale(num_output_channels=1)
            avg_rgb = to_gs(rgb)
            color_dx = torch.abs(avg_rgb[:,:,:,1:] - avg_rgb[:,:,:,:-1])
            color_dy = torch.abs(avg_rgb[:,:,1:,:] - avg_rgb[:,:,:-1,:])

            # weighting
            weight_x = torch.exp(-alpha * color_dx)
            weight_y = torch.exp(-alpha * color_dy)

            # combine
            loss_x = (depth_dx * weight_x).mean()
            loss_y = (depth_dy * weight_y).mean()
            
            logging.debug(f'edge-aware gradient loss: {loss_x+loss_y}')

            loss = loss + 0.2 * (loss_x + loss_y)


        # Check for NaNs or Infs
        if torch.isinf(loss) or torch.isnan(loss):
            logging.error("Abnormal loss (Inf or NaN). Returning 0.")
            return torch.tensor(0.0, device=loss.device)
        
        return loss


###############################################################################
#                    Training & Validation Routines                           #
###############################################################################


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_fn: MonoDepthLoss,
    debug: bool = False,
    profiler=None
) -> float:
    """
    Train the model for a single epoch.

    Args:
        model (nn.Module): The DPT depth model.
        loader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer for the model.
        device (torch.device): GPU or CPU device.
        epoch (int): Current epoch number (for logging).
        loss_fn (MonoDepthLoss): Custom depth loss function.
        debug (bool): Whether to log debug info.
        profiler: (optional) PyTorch profiler context manager if profiling is enabled.

    Returns:
        float: Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    t_start = time.time()

    for batch_idx, data in enumerate(loader):
        rgb = data['rgb'].to(device)
        gt_depth = data['gt'].to(device)
        mask = data.get('mask', None)
        if mask is not None:
            mask = mask.to(device)

        if profiler:
            with record_function("train_batch"):
                pred_depth = model(rgb)
                loss = loss_fn(pred_depth, gt_depth, rgb=rgb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            pred_depth = model(rgb)
            loss = loss_fn(pred_depth, gt_depth, rgb=rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        if debug and (batch_idx % 10 == 0):
            logging.info(
                f"[Epoch {epoch}] Train Batch {batch_idx}/{len(loader)} | "
                f"Loss: {loss.item():.4f}"
            )

        # Step the profiler each iteration
        if profiler:
            profiler.step()

        # Quick debug saves every 10 steps
        if debug and (batch_idx % 10 == 0):
            save_depth(pred_depth[0, 0].detach().cpu().numpy(), 'tmp/color_output.png')
            save_depth(gt_depth[0, 0].detach().cpu().numpy(),     'tmp/color_gt.png')
            save_rgb(rgb[0].detach().cpu().numpy(),               'tmp/input_rgb.png')

    avg_loss = total_loss / len(loader)
    t_end = time.time()
    logging.info(
        f"[Epoch {epoch}] Train finished in {t_end - t_start:.2f}s "
        f"| Average loss: {avg_loss:.4f}"
    )

    return avg_loss


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    loss_fn: MonoDepthLoss,
    debug: bool = False
) -> float:
    """
    Validate model for one epoch.

    Args:
        model (nn.Module): The DPT depth model (in eval mode).
        loader (DataLoader): Validation data loader.
        device (torch.device): GPU or CPU device.
        epoch (int): Current epoch number (for logging).
        loss_fn (MonoDepthLoss): Custom depth loss function.
        debug (bool): Whether to log debug info.

    Returns:
        float: Average validation loss for this epoch.
    """
    model.eval()
    total_loss = 0.0
    t_start = time.time()

    for batch_idx, data in enumerate(loader):
        rgb = data['rgb'].to(device)
        gt_depth = data['gt'].to(device)
        mask = data.get('mask', None)
        if mask is not None:
            mask = mask.to(device)

        pred_depth = model(rgb)
        loss = loss_fn(pred_depth, gt_depth, rgb=rgb)
        total_loss += loss.item()

        if debug and (batch_idx % 10 == 0):
            logging.info(
                f"[Epoch {epoch}] Val Batch {batch_idx}/{len(loader)} => "
                f"Loss {loss.item():.4f}"
            )

    avg_loss = total_loss / len(loader)
    t_end = time.time()
    logging.info(
        f"[Epoch {epoch}] Validation finished in {t_end - t_start:.2f}s "
        f"| avg_loss={avg_loss:.4f}"
    )

    return avg_loss


###############################################################################
#                    Main Training Entry (Pipeline)                           #
###############################################################################


class SingleBatchDataset(NYUDepthDataset):
    """
    A wrapper that ensures we always return a single item from the real dataset,
    effectively enabling a single-batch overfitting scenario if you attach it
    to a DataLoader.
    """
    def __init__(self, real_dataset: NYUDepthDataset, index_to_use=0):
        super().__init__(
            data_dir=real_dataset.data_dir,
            mode=real_dataset.mode,
            use_mask=real_dataset.use_mask,
            add_noise=real_dataset.add_noise,
            height=real_dataset.height,
            width=real_dataset.width,
            resize=real_dataset.resize,
        )
        self.real_dataset = real_dataset
        self.index_to_use = index_to_use

    def __len__(self):
        return 100  # Arbitrarily large, we keep reusing the same sample.

    def __getitem__(self, idx):
        return self.real_dataset[self.index_to_use]


def main(config_path: str):
    """
    Main function for training or debugging a DPT model with a YAML config.

    Args:
        config_path (str): Path to the YAML config file.
    """
    Break.start()

    # 1) Load YAML config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2) Setup logging
    log_level_str = cfg.get("logging", {}).get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logger = setup_logger(log_level)
    logger.info(f"Loaded config from {config_path}: {cfg}")

    training_cfg = cfg['training']

    # 3) Device & Debug
    device_str = training_cfg.get("device", "cpu")
    device = torch_pick_device(device_str)
    debug = bool(training_cfg.get("debug", True))
    logger.info(f"Using device: {device}")

    # 4) Create Dataset & Dataloader
    dataset_path = training_cfg['data_path']
    train_dataset = NYUDepthDataset(
        data_dir=dataset_path,
        mode="train",
        use_mask=True,
        add_noise=True,
        height=480,
        width=640,
        resize=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        pin_memory=True
    )

    val_dataset = NYUDepthDataset(
        data_dir=dataset_path,
        mode="val",
        use_mask=False,
        add_noise=False,
        height=480,
        width=640,
        resize=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # -------------------------------------------------------------------------
    # If you want to do single-batch overfit debugging, simply uncomment below:
    #
    # single_batch_dataset = SingleBatchDataset(train_dataset, index_to_use=0)
    # train_loader = DataLoader(single_batch_dataset, batch_size=1, shuffle=False)
    # val_loader   = DataLoader(single_batch_dataset, batch_size=1, shuffle=False)
    # -------------------------------------------------------------------------

    # 5) Create the Model
    model = DPTDepthModel(
        backbone=training_cfg['backbone'],
        non_negative=True,
        invert=False
    )
    model.to(device)

    # 6) Optimizer & Scheduler
    optimizer = get_optimizer(model, training_cfg['optimizer'])
    scheduler = get_scheduler(optimizer, training_cfg['lr_scheduler'])

    # 7) Profiler Setup
    profiling_cfg = cfg.get("profiling", {})
    enable_profiling = profiling_cfg.get("enable", False)
    steps_to_profile = profiling_cfg.get("steps_to_profile", 5)
    export_trace_path = profiling_cfg.get("export_trace", "profile_trace.json")

    # 8) Optionally Freeze the backbone at start
    freeze_epochs = training_cfg.get('warmup_epochs', 0)
    model.freeze_backbone()
    logger.info("Backbone is frozen at the start.")

    # 9) Construct a depth-loss object
    loss_fn = MonoDepthLoss(device=device)

    best_val_loss = float('inf')
    best_train_loss = float('inf')

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(1, training_cfg["epochs"] + 1):
        # Unfreeze backbone after 'warmup_epochs' if specified
        if epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            logger.info(f"Backbone layers unfrozen at epoch {epoch}")

        # Check if we do a profiling pass this epoch
        if enable_profiling and epoch == 1:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=1, active=steps_to_profile, repeat=1
                ),
                on_trace_ready=lambda p: p.export_chrome_trace(export_trace_path),
                record_shapes=True,
                profile_memory=True
            ) as prof:
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    loss_fn=loss_fn,
                    debug=debug,
                    profiler=prof
                )
        else:
            # Standard training epoch
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                loss_fn=loss_fn,
                debug=debug
            )

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        # Validation
        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            loss_fn=loss_fn,
            debug=debug
        )

        # Step the LR scheduler
        scheduler.step(val_loss)

        logging.info(f"[Epoch {epoch}] Previous LR: {scheduler.get_last_lr()}")

        # Track best val
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            # Optionally save checkpoint
            # save_checkpoint(
            #     model_state_dict=best_model_state,
            #     optimizer_state_dict=optimizer.state_dict(),
            #     epoch=epoch,
            #     val_loss=val_loss,
            #     save_dir=training_cfg["output_dir"],
            #     prefix="dpt_mono",
            # )

    # End of training
    logger.info("Training complete.")
    final_model_path = os.path.join(training_cfg["output_dir"], "dpt_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Best training loss: {best_train_loss:.4f}")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")


###############################################################################
#                           Script Entry Point                                #
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} /path/to/train_config.yaml")
        sys.exit(1)

    main(config_path=sys.argv[1])