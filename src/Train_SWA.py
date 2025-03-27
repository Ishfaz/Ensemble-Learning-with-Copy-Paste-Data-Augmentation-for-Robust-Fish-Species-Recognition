from ultralytics import YOLO
import albumentations as A
from ultralytics.utils import LOGGER, colorstr
import torch
import numpy as np
import os
import copy

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
            A.GaussNoise(var_limit=(10, 50), p=0.2),
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
    """Initialize augmentations and SWA model at start of training"""
    # Initialize custom augmentations
    trainer.transform = setup_augmentations()
    LOGGER.info("Custom augmentations initialized")
    
    # Initialize SWA related parameters
    trainer.swa_enabled = True  # Enable SWA
    trainer.swa_start = 260  # Start SWA from this epoch (you can adjust this)
    trainer.swa_lr = 0.01  # SWA learning rate (you can adjust this)
    trainer.swa_c_epochs = 1  # Collection frequency (every epoch)
    trainer.swa_n = 0  # Counter for number of models collected
    
    # Create SWA model (a deep copy of the current model)
    trainer.swa_model = copy.deepcopy(trainer.model)
    LOGGER.info(f"{colorstr('SWA:')} Initialized, will start at epoch {trainer.swa_start}")

def on_train_batch_start(trainer):
    """Apply augmentations to training batch"""
    if not hasattr(trainer, 'transform'):
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
                                targets[targets[:, 0] == idx][:, 1:] = torch.from_numpy(
                                    np.array(transformed['bboxes'])
                                ).to(targets.device)
                                
                    except Exception as e:
                        continue
                        
                # Yield modified batch back to training process
                yield (images, targets) + batch[2:]
                
    except Exception as e:
        LOGGER.warning(f"Batch augmentation error: {e}")
        yield batch

def on_train_epoch_end(trainer):
    """Update SWA model at the end of each epoch if we're past swa_start"""
    if not hasattr(trainer, 'swa_enabled') or not trainer.swa_enabled:
        return
    
    # Only start SWA after specified epoch
    if trainer.epoch + 1 >= trainer.swa_start:
        # Adjust learning rate according to SWA schedule if needed
        if hasattr(trainer, 'swa_lr'):
            for g in trainer.optimizer.param_groups:
                g['lr'] = trainer.swa_lr
        
        # Only update every swa_c_epochs epochs
        if (trainer.epoch + 1 - trainer.swa_start) % trainer.swa_c_epochs == 0:
            # Moving average update
            LOGGER.info(f"{colorstr('SWA:')} Updating model (n={trainer.swa_n+1})")
            with torch.no_grad():
                for swa_param, param in zip(trainer.swa_model.parameters(), trainer.model.parameters()):
                    swa_param.mul_(trainer.swa_n / (trainer.swa_n + 1))
                    swa_param.add_(param.data / (trainer.swa_n + 1))
            
            trainer.swa_n += 1

def on_train_end(trainer):
    """Save the SWA model at the end of training"""
    if not hasattr(trainer, 'swa_enabled') or not trainer.swa_enabled or trainer.swa_n == 0:
        return
    
    LOGGER.info(f"{colorstr('SWA:')} Finalizing model (n={trainer.swa_n})")
    
    # Update batch normalization statistics for SWA model
    LOGGER.info(f"{colorstr('SWA:')} Updating BatchNorm statistics")
    bn_update(trainer)
    
    # Save SWA model in the correct format
    swa_model_path = os.path.join(trainer.save_dir, 'weights', 'swa_model.pt')
    LOGGER.info(f"{colorstr('SWA:')} Saving model to {swa_model_path}")
    
    # Get a reference to the original model's checkpoint
    original_ckpt = None
    best_model_path = os.path.join(trainer.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        original_ckpt = torch.load(best_model_path, map_location='cpu')
    
    if original_ckpt is not None:
        # Create a proper checkpoint with the same structure as YOLO expects
        swa_ckpt = original_ckpt.copy()
        
        # Replace model weights with SWA model weights
        if 'model' in swa_ckpt and hasattr(swa_ckpt['model'], 'state_dict'):
            swa_ckpt['model'] = trainer.swa_model
        elif 'model' in swa_ckpt:
            swa_ckpt['model'] = trainer.swa_model.state_dict()
        else:
            # If original structure is different, create a new structure
            swa_ckpt = {
                'model': trainer.swa_model,
                'optimizer': trainer.optimizer.state_dict(),
                'epoch': trainer.epoch,
                'date': trainer.date,
                'version': trainer.args.version
            }
        
        # Save the model in the correct format
        torch.save(swa_ckpt, swa_model_path)
        LOGGER.info(f"{colorstr('SWA:')} Model saved successfully in YOLO format")
    else:
        # Fallback if we can't access the original checkpoint format
        LOGGER.warning(f"{colorstr('SWA:')} Could not find best.pt to use as template. Using simplified format.")
        
        # Create a simplified checkpoint
        swa_ckpt = {
            'model': trainer.swa_model
        }
        
        # Save the model
        torch.save(swa_ckpt, swa_model_path)
        LOGGER.info(f"{colorstr('SWA:')} Model saved in simplified format")

def bn_update(trainer):
    """Updates the batch normalization statistics for the SWA model"""
    # Store original model state
    model_was_training = trainer.model.training
    swa_model_was_training = trainer.swa_model.training
    
    # Set SWA model to evaluation mode
    trainer.swa_model.eval()
    
    # Reset running stats
    for module in trainer.swa_model.modules():
        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            module.reset_running_stats()
            module.momentum = None  # Use cumulative moving average
            module.train()
    
    # Accumulate new stats
    with torch.no_grad():
        for batch in trainer.train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                images = batch[0]
                # Forward pass to update BN statistics
                trainer.swa_model(images.to(trainer.device))
    
    # Restore original model states
    if model_was_training:
        trainer.model.train()
    if swa_model_was_training:
        trainer.swa_model.train()
    else:
        trainer.swa_model.eval()

# Initialize model and add callbacks
model = YOLO('yolov8m.pt')
model.add_callback("on_train_start", on_train_start)
model.add_callback("on_train_batch_start", on_train_batch_start)
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_train_end", on_train_end)

# Create specific save directory
save_dir = "/cluster/home/ishfaqab/Saithes_prepared/results/8l_filtered_data"
os.makedirs(save_dir, exist_ok=True)
model.info()
results = model.train(
    project=save_dir,
    name="run8m",
    data="/cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/data.yaml", 
    epochs=300,
    imgsz=640,
    batch=32, 
    dropout=0.2,
    augment=True,
    workers=4,  
    exist_ok=True,
    patience=300)

print(f"\nResults are saved in: {os.path.join(save_dir, 'run8m')}")
print(f"SWA model saved in: {os.path.join(save_dir, 'runm', 'weights', 'swa_model.pt')}")