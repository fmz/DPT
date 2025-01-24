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
        log_level (int): logging.DEBUG, logging.INFO, etc.
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger()
    logger.handlers = []  # Clear existing handlers (especially in notebooks or repeated runs)

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
    Example Monocular Depth Loss combining:
      - Masked L1 (or a custom variant)
      - Optional edge-aware smoothness
      - PiQA-based SSIM or other advanced terms if desired.
    """
    def __init__(self, device):
        self.ssim = piqa.SSIM(n_channels=1).to(device)

    def __call__(self,
                 pred: torch.Tensor,
                 target: torch.Tensor,
                 rgb: torch.Tensor = None,
                 mask: torch.Tensor = None) -> torch.Tensor:
        """
        Applies a combined loss to the predicted and target depth maps.

        Args:
            pred (torch.Tensor): Predicted depth, shape [B,1,H,W] or [B,H,W].
            target (torch.Tensor): Ground truth depth, shape [B,1,H,W].
            rgb (torch.Tensor, optional): RGB image for edge-based losses, [B,3,H,W].
            mask (torch.Tensor, optional): Binary mask for valid depth pixels, [B,1,H,W].

        Returns:
            torch.Tensor: A scalar loss value.
        """

        Break.point()

        # Ensure [B,1,H,W]
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        # Crop edges if needed
        pred = pred[:, :, 8:-8, 8:-8]
        target = target[:, :, 8:-8, 8:-8]
        if rgb is not None:
            rgb = rgb[:, :, 8:-8, 8:-8]

        # Example: Masked L1 (you can revert to actual L1, MSE, etc.)
        if mask is not None:
            valid = mask[:, :, 8:-8, 8:-8].bool()
            l1_loss = F.l1_loss(pred[valid], target[valid])
        else:
            l1_loss = F.l1_loss(pred, target)

            # diff = torch.abs(pred - target)
            # diff = torch.pow(diff + 1, 3) + diff + 1 # NOT really L1...
            # l1_loss = torch.mean(diff)

        logging.debug(f'L1 loss: {l1_loss}')

        loss = l1_loss

        # Combine with SSIM
        # We'll do a basic "normalized" approach if pred & target have non-zero ranges
        max_p, min_p = pred.max(), pred.min()
        max_t, min_t = target.max(), target.min()

        if max_p > min_p and max_t > min_t:
            pred_norm = (pred - min_p) / (max_p - min_p)
            targ_norm = (target - min_t) / (max_t - min_t)

            ssim_loss = 1.0 - self.ssim(pred_norm, targ_norm)

            logging.debug(f'SSIM loss: {ssim_loss}')

            loss = loss + 0.2 * ssim_loss

        # Perform edge-aware-smoothness loss if rgb is available
        # if rgb is not None:
        #     alpha = 10.0
        #     # partial derivatives
        #     depth_dx = torch.abs(pred[:,:,:,1:] - pred[:,:,:,:-1])
        #     depth_dy = torch.abs(pred[:,:,1:,:] - pred[:,:,:-1,:])

        #     # Convert to grayscale for gradient computation
        #     to_gs = Grayscale(num_output_channels=1)
        #     avg_rgb = to_gs(rgb)
        #     color_dx = torch.abs(avg_rgb[:,:,:,1:] - avg_rgb[:,:,:,:-1])
        #     color_dy = torch.abs(avg_rgb[:,:,1:,:] - avg_rgb[:,:,:-1,:])

        #     # weighting
        #     weight_x = torch.exp(-alpha * color_dx)
        #     weight_y = torch.exp(-alpha * color_dy)

        #     # combine
        #     loss_x = (depth_dx * weight_x).mean()
        #     loss_y = (depth_dy * weight_y).mean()

        #     logging.debug(f'edge-aware gradient loss: {loss_x+loss_y}')

        #     loss = loss + 0.2 * (loss_x + loss_y)

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
        optimizer (optim.Optimizer): Optimizer.
        device (torch.device): GPU or CPU device.
        epoch (int): Current epoch (for logging).
        loss_fn (MonoDepthLoss): Depth loss.
        debug (bool): Whether to log debug info.
        profiler: PyTorch profiler (optional).

    Returns:
        float: Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        rgb = batch['rgb'].to(device)
        gt_depth = batch['gt'].to(device)
        mask = batch.get('mask', None)
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

        if (batch_idx % 10 == 0):
            logging.info(f"[Epoch {epoch}] Train Batch {batch_idx}/{len(loader)} | "
                         f"Loss: {loss.item():.4f}")

        if profiler:
            profiler.step()

        # Quick debug saves every 10 steps
        if debug and (batch_idx % 1 == 0):
            detached_pred = pred_depth[0, 0].detach().cpu().numpy()
            detached_gt   = gt_depth[0, 0].detach().cpu().numpy()
            detached_rgb  = rgb[0].detach().cpu().numpy()

            detached_pred[:, :8] = 0
            detached_pred[:, -8:] = 0
            detached_gt[:, :8] = 0
            detached_gt[:, -8:] = 0

            detached_pred[:8, :] = 0
            detached_pred[-8:, :] = 0
            detached_gt[:8, :] = 0
            detached_gt[-8:, :] = 0

            save_depth(detached_pred, 'tmp/pred_depth.png')
            save_depth(detached_gt,   'tmp/gt_depth.png')
            save_rgb(detached_rgb,    'tmp/input_rgb.png')

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
    Validate the model for a single epoch.

    Args:
        model (nn.Module): The DPT depth model (eval mode).
        loader (DataLoader): Validation data loader.
        device (torch.device): GPU or CPU device.
        epoch (int): Current epoch (for logging).
        loss_fn (MonoDepthLoss): Depth loss function.
        debug (bool): Whether to log debug info.

    Returns:
        float: Average validation loss for this epoch.
    """
    model.eval()
    total_loss = 0.0
    t_start = time.time()

    for batch_idx, batch in enumerate(loader):
        rgb = batch['rgb'].to(device)
        gt_depth = batch['gt'].to(device)
        mask = batch.get('mask', None)
        if mask is not None:
            mask = mask.to(device)

        pred_depth = model(rgb)
        loss = loss_fn(pred_depth, gt_depth, rgb=rgb)
        total_loss += loss.item()

        if (batch_idx % 10 == 0):
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
#           Single-Batch Overfitting Dataset Wrapper (Optional)               #
###############################################################################

class SingleBatchDataset(NYUDepthDataset):
    """
    Wraps a standard dataset but always returns a single item (index_to_use),
    enabling single-batch overfitting for debugging.
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
        return 100  # Arbitrarily large; we reuse the same sample.

    def __getitem__(self, idx):
        return self.real_dataset[self.index_to_use]


###############################################################################
#                       Fine-Tuning & Main Training Logic                     #
###############################################################################

def load_local_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        device: torch.device,
        strict=False) -> None:
    """
    Loads a local checkpoint into the model's state_dict.

    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): The device for loading.
        strict (bool): Whether to enforce that all keys match exactly.
    """
    logging.info(f"Attempting to load checkpoint from {checkpoint_path} with strict={strict}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # If the checkpoint was saved with a dictionary containing "model_state" or similar
    if "model_state" in ckpt:
        model_sd = ckpt["model_state"]
    else:
        model_sd = ckpt  # assume it's a direct state_dict

    missing, unexpected = model.load_state_dict(model_sd, strict=strict)
    if missing:
        logging.warning(f"Missing keys in state_dict: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys in state_dict: {unexpected}")
    logging.info("Checkpoint loaded.")


def main(config_path: str):
    """
    Main function for training/fine-tuning a DPT model with a YAML config.

    Args:
        config_path (str): Path to the YAML config file.
    """
    Break.start()

    # 1) Load YAML
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2) Logging
    log_level_str = cfg.get("logging", {}).get("level", "INFO").upper()
    logger = setup_logger(getattr(logging, log_level_str, logging.INFO))
    logger.info(f"Loaded config from {config_path}: {cfg}")

    training_cfg = cfg['training']
    debug = bool(training_cfg.get('debug', False))

    # 3) Device
    device_str = training_cfg.get("device", "cpu")
    device = torch_pick_device(device_str)
    logging.info(f"Using device: {device}")

    # 4) Dataset & Dataloader
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

    # # Uncomment for single-batch overfitting
    # single_batch_dataset = SingleBatchDataset(train_dataset, index_to_use=0)
    # train_loader = DataLoader(single_batch_dataset, batch_size=1, shuffle=False)
    # val_loader   = DataLoader(single_batch_dataset, batch_size=1, shuffle=False)

    # 5) Create/Load Model
    backbone_name = training_cfg.get("backbone", "vitb16_384")
    model = DPTDepthModel(
        backbone=backbone_name,
        non_negative=True,
        invert=True
    )
    model.to(device)

    # Check if we want to load a local checkpoint
    local_ckpt = training_cfg.get("pretrained_model_path", None)
    if local_ckpt and os.path.isfile(local_ckpt):
        logging.info(f"Loading local checkpoint: {local_ckpt}")
        load_local_checkpoint(model, local_ckpt, device=device, strict=False)
        out_model_prefix = "dpt_mono_finetune"
    else:
        if local_ckpt:
            logging.warning(f"model_path={local_ckpt} not found, using default timm-based weights.")

        out_model_prefix = "dpt_mono"

    # 6) Freeze/Unfreeze Setup
    freeze_until = training_cfg.get("freeze_backbone_until", 0)
    if freeze_until != 0:
        logging.info("Freezing the backbone for partial fine-tuning.")
        model.freeze_backbone()

    # 7) Optimizer & Scheduler
    optimizer = get_optimizer(model, training_cfg['optimizer'])
    scheduler = get_scheduler(optimizer, training_cfg['lr_scheduler'])

    # 8) Profiling
    profiling_cfg = cfg.get("profiling", {})
    enable_profiling = profiling_cfg.get("enable", False)
    steps_to_profile = profiling_cfg.get("steps_to_profile", 5)
    export_trace_path = profiling_cfg.get("export_trace", "profile_trace.json")

    # 9) Construct the DepthLoss
    loss_fn = MonoDepthLoss(device=device)

    # 10) Training Loop
    best_val_loss = float('inf')
    best_model_state = None
    epochs = training_cfg.get("epochs", 10)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        # Optionally unfreeze backbone after warmup
        if epoch == freeze_until + 1:
            logging.info(f"Unfreezing the backbone at epoch {epoch}")
            model.unfreeze_backbone()

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

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            loss_fn=loss_fn,
            debug=debug
        )

        scheduler.step(val_loss)
        logging.info(f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

            logging.info(f"New best val_loss={best_val_loss:.4f}")

            if (cfg.get("save_intermediate_models", False)):
                save_checkpoint(
                    model_state_dict=best_model_state,
                    optimizer_state_dict=optimizer.state_dict(),
                    epoch=epoch,
                    val_loss=val_loss,
                    save_dir=training_cfg["output_dir"],
                    prefix=out_model_prefix,
                )

    # Final Save
    final_model_path = os.path.join(training_cfg["output_dir"], "dpt_final_finetuned.pth")
    os.makedirs(training_cfg["output_dir"], exist_ok=True)
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
