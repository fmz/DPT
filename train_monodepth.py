import time
import copy
import yaml
import logging
import os
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision.transforms import Grayscale

import piqa

from dataset.nyuloader_v2 import NYUDepthDataset
from dpt.models import DPTDepthModel

from debug import Break

from utils import (
    get_optimizer,
    save_depth,
    save_rgb,
    save_checkpoint,
    torch_pick_device,
    get_scheduler
)

class SingleBatchDataset(NYUDepthDataset):
    '''
    A wrapper that takes a 'real' dataset, but returns only the first item every time.
    or returns a user-chosen index. 
    '''

    def __init__(self, real_dataset, index_to_use=0):
        self.real_dataset = real_dataset
        self.index_to_use = index_to_use

    def __len__(self):
        # We can artificially set this to e.g. 100, but it doesn't matter
        return 100  

    def __getitem__(self, idx):
        # Always return the same item
        return self.real_dataset[self.index_to_use]
        

def setup_logger(log_level=logging.INFO):
    ''' Configure root logger format and level '''
    logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.setLevel(log_level)
    logger.addHandler(handler)

    return logger

ssim = piqa.SSIM(n_channels=1).cuda()

def masked_l1_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        rgb: torch.Tensor = None,
        mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Masked L1 Loss.
    pred/target shape: [B, 1, H, W]
    mask shape: [B, 1, H, W] with True=valid.
    """
    
    Break.point()

    # Trim unusable parts
    pred   = pred[:, :, 8:-8, 8:-8]
    target = target[:, :, 8:-8, 8:-8]

    if mask is not None:
        valid = mask.bool()
        l1_loss = F.l1_loss(pred[valid], target[valid])
    else:
        l1_loss = F.l1_loss(pred, target)

    logging.info(f"L1 loss = {l1_loss}")

    loss = l1_loss 

    global ssim
    max_p = pred.max()
    max_t = target.max()
    if max_p != 0 and max_t != 0:
        min_p = pred.min()
        min_t = target.min()
        
        pred_normalized = (pred - min_p) / (max_p - min_p)
        target_normalized = (target - min_t) / (max_t - min_t)
        ssim_loss = 1.0 - ssim(pred_normalized, target_normalized)
        
        logging.info(f"Relative SSIM loss = {ssim_loss}")

        loss = loss * 0.5 + ssim_loss * 0.5

    # if rgb is not None:
    #     # Compute gradient loss
    #     pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    #     pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

    #     rgb_dx = torch.abs(rgb[:, :, :, 1:] - rgb[:, :, :, :-1])
    #     rgb_dy = torch.abs(rgb[:, :, 1:, :] - rgb[:, :, :-1, :])
        
    #     to_gs = Grayscale(num_output_channels=1)
    #     gs_dx = to_gs(rgb_dx)
    #     gs_dy = to_gs(rgb_dy)

    #     err_x = torch.abs(pred_dx - gs_dx)
    #     err_y = torch.abs(pred_dy - gs_dy)

    #     grad_loss = torch.mean(err_x) + torch.mean(err_y)
    #     logging.debug(f'Gradient loss: {grad_loss}')

        # save_depth(pred_dx[0,0].detach().cpu().numpy(), 'tmp/pred_dx.png')
        # save_depth(pred_dy[0,0].detach().cpu().numpy(), 'tmp/pred_dy.png')

        # save_depth(gs_dx[0,0].detach().cpu().numpy(), 'tmp/gs_dx.png')
        # save_depth(gs_dy[0,0].detach().cpu().numpy(), 'tmp/gs_dy.png')

        # save_depth(err_x[0,0].detach().cpu().numpy(), 'tmp/err_x.png')
        # save_depth(err_y[0,0].detach().cpu().numpy(), 'tmp/err_y.png')

    #    loss += grad_loss * 0.1 # TODO: un-hardcode weight

    if torch.isinf(loss):
        logging.error("Infinite loss detected, stopping.")
        return 0
    if torch.isnan(loss):
        logging.error("NaN loss detected, stopping.")
        return 0
    return loss

def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        epoch: int,
        debug: bool,
        profiler=None
) -> float:
    '''
    Train the model for 1 epoch.

    Args:
        Args:
        model (nn.Module): The DPT depth model.
        loader (DataLoader): Training data loader.
        optimizer (optim.Optimizer): Optimizer for model.
        device (torch.device): GPU or CPU device.
        epoch (int): Current epoch number (for logging).
        debug (bool): Whether to log debug info.
        profiler: PyTorch profiler context manager if profiling is enabled.

    Returns:
        float: Average training loss for this epoch.
    '''

    model.train()
    total_loss = 0.0
    t_start = time.time()

    for batch_idx, data in enumerate(loader):
        rgb = data['rgb'].to(device)
        gt_depth = data['gt'].to(device)

        if profiler is not None:
            with record_function("train_batch"):
                pred_depth = model(rgb)
                loss = masked_l1_loss(pred_depth, gt_depth, rgb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            pred_depth = model(rgb)
            loss = masked_l1_loss(pred_depth, gt_depth, rgb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()

        if debug and (batch_idx % 10 == 0):
            logging.info(f'[Epoch {epoch}], Train Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}')

        if profiler is not None:
            profiler.step()

        if (batch_idx % 10 == 0):
            save_depth(pred_depth[0, 0].detach().cpu().numpy(), 'tmp/color_output.png')
            save_depth(gt_depth[0, 0].detach().cpu().numpy(),     'tmp/color_gt.png')
            save_rgb(rgb[0].detach().cpu().numpy(),          'tmp/input_rgb.png')

    avg_loss = total_loss / len(loader)
    t_end = time.time()
    logging.info(f'[Epoch {epoch}] Finished in {(t_start - t_end):.2f}s | Average loss: {avg_loss:.4f}')

    return avg_loss

@torch.no_grad()
def validate_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        epoch: int,
        debug: bool
) -> float:
    '''
    Validate model for one epoch.

    Args:
        model (nn.Module): The DPT depth model.
        loader (DataLoader): Validation data loader.
        device (torch.device): GPU or CPU device.
        epoch (int): Current epoch number (for logging).
        debug (bool): Whether to log debug info.

    Returns:
        float: Average validation loss for this epoch.

    '''
    model.eval()
    total_loss = 0.0

    for batch_idx, data in enumerate(loader):
        rgb = data['rgb'].to(device)
        gt_depth = data['gt'].to(device)

        pred_depth = model(rgb)
        loss = masked_l1_loss(pred_depth, gt_depth, rgb)
        total_loss += loss.item()

        if debug and batch_idx % 10 == 0:
            logging.info(f"[Epoch {epoch}] Val Batch {batch_idx}/{len(loader)} => loss {loss.item():.4f}")
        
    avg_loss = total_loss / len(loader)
    logging.info(f'[Epoch {epoch}] Validation complete | avg_loss={avg_loss:.4f}')
    
    return avg_loss

def main(config_path: str):
    '''
    Main function for training DPT with a YAML config.

    Args:
        config_path (str): Path to the YAML config file.
    '''

    Break.start()

    # Load YAML config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Setup logging
    log_level_str = cfg.get("logging", {}).get("level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    setup_logger(log_level)

    logging.info(f'Loaded config from {config_path}: {cfg}')

    training_cfg = cfg['training']
    device_str = training_cfg.get("device", "cpu")
    device = torch_pick_device(device_str)
    logging.info(f'Using device: {device}')

    debug = bool(training_cfg.get("debug", True))

    # Create Datasets & DataLoaders (NYU example)
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
        pin_memory=True,
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
    
    ### Overfitting stuff

    # Wrap it so we always return item #0
    # single_batch_dataset = SingleBatchDataset(train_dataset, index_to_use=0)

    # single_batch_loader = DataLoader(
    #     single_batch_dataset,
    #     batch_size=1,       # or 2, but keep it small
    #     shuffle=False,      # no shuffling
    #     num_workers=0,      # reduce concurrency to keep it simple
    #     drop_last=False
    # )

    # train_loader = single_batch_loader
    # val_loader   = single_batch_loader

    ### End overfitting stuff

    # Create model
    model = DPTDepthModel(
        backbone=training_cfg['backbone'],
        non_negative=True,
        invert=False
    )
    model.to(device)
    #model = model.compile()

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, training_cfg['optimizer'])
    scheduler = get_scheduler(optimizer, training_cfg['lr_scheduler'])

    # Setup profiler
    profiling_cfg = cfg.get("profiling", {})
    enable_profiling = profiling_cfg.get("enable", False)
    steps_to_profile = profiling_cfg.get("steps_to_profile", 5)
    export_trace_path = profiling_cfg.get("export_trace", "profile_trace.json")

    # Training loop
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_model_state = None

    # Start with the backbone frozen
    model.freeze_backbone()

    for epoch in range(1, training_cfg["epochs"]+1):
        # Unfreeze backbone at some point
        if epoch == training_cfg['warmup_epochs']:
            model.unfreeze_backbone()

        # If profiling is enabled, run with pytorch profiler context
        if enable_profiling and epoch == 1:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA, ProfilerActivity.XPU],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=1, active=steps_to_profile, repeat=1
                ),
                on_trace_ready=lambda p: p.export_trace_path(export_trace_path),
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch,
                    debug=debug,
                    profiler=prof,
                )
        else:
            # No profiler
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                debug=debug,
                profiler=None,
            )
        if train_loss < best_train_loss:
            best_train_loss = train_loss

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            debug=debug,
        )
        scheduler.step(val_loss)
        logging.info(f'[Epoch {epoch}], prev LR: {scheduler.get_last_lr()}')


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            # save_checkpoint(
            #     model_state_dict=best_model_state,
            #     optimizer_state_dict=optimizer.state_dict(),
            #     epoch=epoch,
            #     val_loss=val_loss,
            #     save_dir=training_cfg["output_dir"],
            #     prefix="dpt_mono",
            # )
    # End of training
    logging.info("Training complete.")
    final_model_path = os.path.join(training_cfg["output_dir"], "dpt_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved to {final_model_path}")
    logging.info(f"Best training loss: {best_train_loss:.4f}")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} /path/to/train_config.yaml")
        sys.exit(1)

    main(config_path=sys.argv[1])