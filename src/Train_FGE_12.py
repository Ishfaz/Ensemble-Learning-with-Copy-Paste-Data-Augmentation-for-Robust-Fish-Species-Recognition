import argparse
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO
import logging
from pathlib import Path
import albumentations as A
from ultralytics.utils import LOGGER, colorstr
import math
import json

def setup_augmentations(p=1.0):
    """Setup custom augmentation pipeline"""
    prefix = colorstr("Custom albumentations: ")
    try:
        T = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
            A.MotionBlur(blur_limit=7, p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, p=0.3),
            A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=20, p=0.3),
            A.GaussNoise(var=(10, 50), p=0.2),  # Changed from var_limit to var
            A.RandomGamma(gamma_limit=(70,130), p=0.3),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomSizedBBoxSafeCrop(height=640, width=640, erosion_rate=0.2, p=0.3),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.3)
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
        # Access batch through train_loader iterator
        for batch in trainer.train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
                
                # Apply transformations
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
                            
                            images[idx] = torch.from_numpy(
                                transformed['image']
                            ).permute(2, 0, 1).to(images.device)
                            
                            if transformed['bboxes']:
                                # Fix the invalid slice syntax
                                targets[targets[:, 0] == idx, 1:] = torch.from_numpy(
                                    np.array(transformed['bboxes'])
                                ).to(targets.device)
                    except Exception as e:
                        continue
                        
                # Yield modified batch back to training process
                yield (images, targets) + batch[2:]
                
    except Exception as e:
        LOGGER.warning(f"Batch augmentation error: {e}")
        yield batch

def get_fge_lr(current_epoch, cycle_length, lr_min, lr_max):
    """
    'Fast Geometric Ensembling' style triangular schedule.
    - position = (current_epoch % cycle_length) / cycle_length in [0,1)
    - position=0 -> LR=lr_max
    - position=0.5 -> LR=lr_min
    - position=1.0 -> LR=lr_max
    """
    position = (current_epoch % cycle_length) / cycle_length
    if position < 0.5:
        # Descending from lr_max down to lr_min
        frac = position / 0.5  # goes from 0->1 as we move 0->0.5
        lr = lr_max - (lr_max - lr_min) * frac
    else:
        # Ascending from lr_min back to lr_max
        frac = (position - 0.5) / 0.5  # goes from 0->1 as we move 0.5->1.0
        lr = lr_min + (lr_max - lr_min) * frac
    return round(lr, 7)

def is_snapshot_point(current_epoch, cycle_length):
    # Adjust epoch to start counting from FGE phase
    adjusted_epoch = current_epoch - 260  # Since FGE starts at base_epoch+1
    
    # Calculate position in cycle (0 to cycle_length-1)
    cycle_position = adjusted_epoch % cycle_length
    
    # For any cycle length, the minimum LR is at half-way point
    # For cycle_length=4: positions are 0,1,2,3 and we want 1
    # For cycle_length=2: positions are 0,1 and we want 0
    snapshot_position = (cycle_length // 2) - 1
    return cycle_position == snapshot_position

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

def main(args):
    """Main training function"""
    # Create necessary directories
    os.makedirs(args.save_dir, exist_ok=True)
    logger = setup_training_logger(args.save_dir)
    logger.info(f"Starting training with args: {args}")
    
    base_training_dir = Path(args.save_dir) / 'base_training'
    weights_dir = base_training_dir / 'weights'
    base_training_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = YOLO(args.model_path)
    
    # Add augmentation callbacks
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_start", on_train_batch_start)
    
    # ------------------------ Base training phase ------------------------ #
    if args.base_epoch > 0:
        logger.info(f"Starting base training for {args.base_epoch} epochs")
        results = model.train(
            data=args.data_path,
            epochs=args.base_epoch,  # Train till epoch 350
            batch=args.batch_size,
            imgsz=args.image_size,
            device=args.device,
            project=str(args.save_dir),
            name='base_training',
            exist_ok=True,
            workers=args.workers,
            save=True,
            save_dir=str(base_training_dir),
            patience=args.patience,
            augment=True,
            optimizer=args.optimizer.lower()  # Make sure it's lowercase
        )
        
        base_ckpt = weights_dir / 'last.pt'
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Base training checkpoint not found at {base_ckpt}")
        model = YOLO(str(base_ckpt))
        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_batch_start", on_train_batch_start)
        logger.info(f"Base model loaded from {base_ckpt}")
    
    # ------------------------ FGE cyclical phase ------------------------ #
    remaining_epochs = args.epochs - args.base_epoch
    if remaining_epochs > 0:
        logger.info(f"Starting FGE cyclical phase from epoch {args.base_epoch + 1}")
        
        fge_dir = Path(args.save_dir) / 'FGE_ensemble'
        fge_weights_dir = fge_dir / 'weights'
        metrics_dir = fge_dir / 'metrics'
        fge_dir.mkdir(parents=True, exist_ok=True)
        fge_weights_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        cycle_length = args.cycle  # Should be 4
        num_cycles = math.ceil(remaining_epochs / cycle_length)
        ensemble_results = []
        
        global_epoch = args.base_epoch + 1  # Start from epoch 351
        
        for cycle_idx in range(num_cycles):
            # Number of epochs we actually do this cycle
            # (in case the final cycle is truncated)
            cycle_epochs = min(cycle_length, remaining_epochs - cycle_idx * cycle_length)
            
            for e in range(cycle_epochs):
                current_epoch = global_epoch + e
                current_lr = get_fge_lr(current_epoch, cycle_length, args.base_lr, args.max_lr)
                
                logger.info(f"FGE Cycle {cycle_idx}, Epoch {current_epoch}, LR={current_lr}")
                logger.info(f"Is snapshot point: {is_snapshot_point(current_epoch, cycle_length)}")
                
                # Train for 1 epoch with that LR
                results = model.train(
                    data=args.data_path,
                    epochs=1,
                    batch=args.batch_size,
                    imgsz=args.image_size,
                    device=args.device,
                    project=str(args.save_dir),
                    name=f'fge_cycle_{cycle_idx}_epoch_{current_epoch}',
                    exist_ok=True,
                    workers=args.workers,
                    lr0=float(current_lr),
                    lrf=float(current_lr),  # keep it constant for that single epoch
                    save=False,  # we do our own snapshot
                    augment=True,  # IMPORTANT: Keep augmentation on
                    save_dir=str(fge_weights_dir)
                )
                
                # If we are at the middle of this cycle => minimal LR => snapshot!
                if is_snapshot_point(current_epoch, cycle_length):
                    model_path = fge_weights_dir / f'fge_model_epoch_{current_epoch}.pt'
                    model.save(str(model_path))
                    
                    # Evaluate on val set
                    logger.info(f"Taking FGE snapshot at epoch {current_epoch}")
                    cycle_pos = (current_epoch % cycle_length) / cycle_length
                    
                    # Run validation
                    try:
                        val_results = model.val(data=args.data_path)  # Change from validate() to val()
                        metrics = {
                            'epoch': current_epoch,
                            'cycle_position': cycle_pos,
                            'lr': current_lr,
                            'mAP50': float(val_results.box.map50),
                            'mAP50-95': float(val_results.box.map),
                            'precision': float(val_results.box.mp),
                            'recall': float(val_results.box.mr)
                        }
                        # Save metrics to JSON
                        metrics_file = metrics_dir / f'metrics_epoch_{current_epoch}.json'
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                    except Exception as e:
                        logger.error(f"Validation error: {e}")

                    if val_results:
                        metrics = {
                            'epoch': current_epoch,
                            'cycle_position': cycle_pos,
                            'lr': current_lr,
                            'mAP50': float(val_results.box.map50),
                            'mAP50-95': float(val_results.box.map),
                            'precision': float(val_results.box.mp),
                            'recall': float(val_results.box.mr)
                        }
                    
                        metrics_file = metrics_dir / f'metrics_epoch_{current_epoch}.json'
                        with open(metrics_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                    
                        logger.info(
                            f"[FGE] Snapshot at epoch {current_epoch}: LR={current_lr}, mAP50={metrics['mAP50']}"
                        )
                    
                        ensemble_results.append({
                            'epoch': int(current_epoch),
                            'model_path': str(model_path),
                            'metrics': metrics,
                            'cycle': int(cycle_idx),
                            'lr': float(current_lr)
                        })
            
            global_epoch += cycle_epochs
        
        # Store final summary with metrics
        if ensemble_results:
            summary_file = fge_dir / 'ensemble_summary.json'
            best_model = max(ensemble_results, key=lambda x: x['metrics']['mAP50'])
            
            with open(summary_file, 'w') as f:
                json.dump({
                    'ensemble_models': ensemble_results,
                    'best_model': best_model
                }, f, indent=2)
            
            logger.info(f"FGE complete. Total snapshots: {len(ensemble_results)}")
            
            # Pick top 5 by mAP50 for ensembling
            best_snapshots = sorted(
                ensemble_results, key=lambda x: x['metrics']['mAP50'], reverse=True
            )[:5]
            snapshot_paths = [r['model_path'] for r in best_snapshots]
            logger.info(f"Top 5 snapshot paths: {snapshot_paths}")
            
            # Create final ensemble model
            logger.info("Creating final ensemble model from top snapshots")
            # Load the models
            models = [YOLO(path) for path in snapshot_paths[:3]]  # Use top 3 for ensemble
            
            # Final validation to demonstrate the ensemble
            logger.info("Running validation on individual models and ensemble:")
            for i, m in enumerate(models):
                # Final validation using proper method
                try:
                    val = m.validate(data=args.data_path)
                    if val and hasattr(val.box, 'map50'):
                        logger.info(f"Model {i} mAP50: {val.box.map50:.4f}")
                except Exception as e:
                    logger.error(f"Validation error for model {i}: {e}")
        else:
            logger.warning("No FGE snapshots were saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FGE training for YOLOv8 with custom augmentations')
    
    parser.add_argument('--data_path', type=str, required=True,
                        help='path to data.yaml file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='path to initial model (e.g., yolov8l.pt)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='total number of epochs (base + FGE) to train')
    parser.add_argument('--base_epoch', type=int, default=340,
                        help='number of epochs for base training')
    parser.add_argument('--cycle', type=int, default=4,
                        help='length of each FGE cycle in epochs (should be even)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size')
    parser.add_argument('--image_size', type=int, default=640,
                        help='input image size')
    parser.add_argument('--base_lr', type=float, default=0.0001,
                        help='lower learning rate bound for FGE cycles')
    parser.add_argument('--max_lr', type=float, default=0.01,
                        help='upper learning rate bound for FGE cycles')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam', 'adamw'], 
                        default='sgd', help='optimizer to use (sgd, adam, or adamw)')
    parser.add_argument('--device', type=str, default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of worker threads')
    parser.add_argument('--patience', type=int, default=340,
                        help='early stopping patience (epochs)')
    parser.add_argument('--augment', action='store_true',
                        help='enable data augmentation')  # Add this line
    
    args = parser.parse_args()
    
    # For reproducibility
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    
    main(args)