import argparse
import numpy as np
import os
import sys
import time
import torch
from ultralytics import YOLO
import logging
from pathlib import Path
import albumentations as A
from ultralytics.utils import LOGGER, colorstr
import json
import traceback

def setup_augmentations(p=1.0):
    """Setup custom augmentation pipeline"""
    prefix = colorstr("Custom albumentations: ")
    try:
        T = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.RandomGamma(gamma_limit=(70,130), p=0.3),
            A.RandomSizedBBoxSafeCrop(height=640, width=640, erosion_rate=0.2, p=0.3),
        ]
        LOGGER.info(f"{prefix}" + ", ".join(f"{x.__class__.__name__}(p={x.p})" for x in T))
        return A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    except Exception as e:
        LOGGER.warning(f"{prefix}Error: {e}")
        return None

def on_train_start(trainer):
    """Initialize augmentations at start of training"""
    trainer.transform = setup_augmentations()
    LOGGER.info("Custom augmentations initialized")

def on_train_batch_start(trainer):
    """Apply augmentations to training batch"""
    if not hasattr(trainer, 'transform') or trainer.transform is None:
        return
    try:
        batch = trainer.data
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images, targets = batch[0].clone(), batch[1].clone()
            for idx in range(len(images)):
                try:
                    img = images[idx].permute(1, 2, 0).cpu().numpy()
                    boxes = targets[targets[:, 0] == idx][:, 1:].cpu().numpy()
                    if len(boxes) > 0:
                        transformed = trainer.transform(
                            image=img,
                            bboxes=boxes,
                            class_labels=['0'] * len(boxes)
                        )
                        images[idx] = torch.from_numpy(transformed['image']).permute(2, 0, 1).to(images.device)
                        if transformed['bboxes']:
                            target_mask = targets[:, 0] == idx
                            targets[target_mask, 1:] = torch.from_numpy(np.array(transformed['bboxes'])).to(targets.device)
                except Exception as e:
                    LOGGER.warning(f"Transform error on image {idx}: {e}")
                    continue
            trainer.batch = (images, targets) + batch[2:]
    except Exception as e:
        LOGGER.warning(f"Batch augmentation error: {e}")

def setup_training_logger(save_dir):
    """Setup detailed logging configuration"""
    log_file = os.path.join(save_dir, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def update_swa_model(swa_model, model, swa_n):
    """Update SWA model with simple average of collected weights"""
    try:
        current_device = next(model.model.parameters()).device
        swa_device = next(swa_model.model.parameters()).device
        
        if swa_device != current_device:
            swa_model.model.to(current_device)
        
        with torch.no_grad():
            for swa_param, param in zip(swa_model.model.parameters(), model.model.parameters()):
                swa_param.data = (swa_param.data * swa_n + param.data) / (swa_n + 1)
        
        return swa_model, swa_n + 1
    except Exception as e:
        LOGGER.error(f"Error in update_swa_model: {e}")
        raise

def bn_update(train_loader, swa_model, device="cuda"):
    """Update BatchNorm statistics for SWA model using training data"""
    LOGGER.info("Updating BatchNorm statistics for SWA model...")
    swa_model.model.train()
    
    # Move model to correct device
    model_device = next(swa_model.model.parameters()).device
    if str(model_device) != str(device):
        swa_model.model.to(device)
    
    # Temporarily set momentum to 1.0 for exact average calculation
    momenta = {}
    for module in swa_model.model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            momenta[module] = module.momentum
            module.momentum = 1.0  # Use full batch statistics
    
    # Process training batches
    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 100:  # Use first 100 batches
                break
                
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images = batch[0].to(device)
                _ = swa_model.model(images)
            
            if batch_idx % 10 == 0:
                LOGGER.info(f"BN update progress: {batch_idx}/100 batches processed")
    
    # Restore original momentum values
    for module in momenta.keys():
        module.momentum = momenta[module]

def schedule(epoch, swa_start, swa_lr, lr_init):
    """Learning rate schedule with constant SWA phase"""
    if epoch < swa_start:
        t = epoch / swa_start
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - swa_lr/lr_init) * (t - 0.5) / 0.4
        else:
            factor = swa_lr/lr_init
        return lr_init * factor
    else:
        return swa_lr

def save_checkpoint(save_dir, epoch, model, swa_model=None, swa_n=None):
    """Save training state with proper device handling"""
    try:
        state = {
            'epoch': epoch,
            'model': model.model.state_dict(),
            'swa_model': swa_model.model.state_dict() if swa_model else None,
            'swa_n': swa_n
        }
        
        checkpoint_path = Path(save_dir) / f'checkpoint_{epoch}.pt'
        torch.save(state, checkpoint_path)
        LOGGER.info(f"Saved checkpoint to {checkpoint_path}")
    except Exception as e:
        LOGGER.error(f"Error saving checkpoint: {e}")

def main(args):
    # Initialization
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    global LOGGER
    LOGGER = setup_training_logger(args.save_dir)
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device setup
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != 'cpu' else "cpu")
    LOGGER.info(f"Using device: {device}")

    # Model initialization
    model = YOLO(args.model_path).to(device)
    swa_model = YOLO(args.model_path).to(device) if args.swa else None
    swa_n = 0

    # Resume training
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.model.load_state_dict(checkpoint['model'])
            if args.swa and checkpoint.get('swa_model'):
                swa_model.model.load_state_dict(checkpoint['swa_model'])
                swa_n = checkpoint.get('swa_n', 0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            LOGGER.info(f"Resumed training from epoch {start_epoch}")
        except Exception as e:
            LOGGER.error(f"Failed to resume training: {e}")

    # Training loop
    best_map = 0.0
    for epoch in range(start_epoch, args.epochs):
        LOGGER.info(f"\n{'='*30} Epoch {epoch+1}/{args.epochs} {'='*30}")
        
        # Set learning rate
        lr = schedule(epoch, args.swa_start, args.swa_lr, args.lr_init)
        LOGGER.info(f"Current learning rate: {lr:.6f}")

        # Configure augmentations
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_batch_start", on_train_batch_start)

        # Train epoch
        results = model.train(
            data=args.data_path,
            epochs=epoch+1,
            batch=args.batch_size,
            imgsz=args.image_size,
            device=device,
            workers=args.workers,
            lr0=lr,
            augment=args.augment,
            project=args.save_dir,
            exist_ok=True,
            pretrained=False,
            resume=False
        )

        # Update SWA model
        if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
            swa_model, swa_n = update_swa_model(swa_model, model, swa_n)
            LOGGER.info(f"SWA updated with {swa_n} models averaged")

        # Validation
        if (epoch + 1) % args.eval_freq == 0:
            val_results = model.val()
            current_map = val_results.box.map50
            LOGGER.info(f"Validation mAP50: {current_map:.4f}")

            # Update best model
            if current_map > best_map:
                best_map = current_map
                model.save(Path(args.save_dir) / "best.pt")
                LOGGER.info(f"New best model saved with mAP50: {best_map:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(args.save_dir, epoch+1, model, swa_model, swa_n)

    # Final SWA processing
    if args.swa and swa_n > 0:
        try:
            # Update BN statistics
            train_loader = model.trainer.train_loader
            bn_update(train_loader, swa_model, device)
            
            # Validate SWA model
            swa_results = swa_model.val()
            swa_map = swa_results.box.map50
            LOGGER.info(f"Final SWA model mAP50: {swa_map:.4f}")
            
            # Save final models
            swa_model.save(Path(args.save_dir) / "swa_final.pt")
            model.save(Path(args.save_dir) / "final.pt")
        except Exception as e:
            LOGGER.error(f"Final SWA processing failed: {e}")

    LOGGER.info("Training completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 SWA Training")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--swa', action='store_true')
    parser.add_argument('--swa_start', type=int, default=160)
    parser.add_argument('--swa_lr', type=float, default=0.05)
    parser.add_argument('--swa_c_epochs', type=int, default=1)
    parser.add_argument('--lr_init', type=float, default=0.01)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=25)
    parser.add_argument('--resume', type=str)
    
    args = parser.parse_args()
    main(args)