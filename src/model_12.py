from ultralytics import YOLO
import albumentations as A
from ultralytics.utils import LOGGER, colorstr
import torch
import numpy as np
import os

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
            A.GaussNoise(var_limit=(10, 50), p=0.2)
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

# Initialize model and add callbacks
save_dir = "/cluster/home/ishfaqab/Saithes_prepared/results/orginal_12l_prep_data"
os.makedirs(save_dir, exist_ok=True)
model = YOLO('yolo12m.pt')
model.add_callback("on_train_start", on_train_start)
model.add_callback("on_train_batch_start", on_train_batch_start)

# Configure training
model.info()
results = model.train(
    project=save_dir,
    name="run12mnew",
    data="/cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/data.yaml", 
    epochs=300,
    imgsz=640,
    batch=32, 
    dropout=0.2,
    augment=True,
    workers=4,  
    exist_ok=True,
    patience=300
)

print(f"\nResults are saved in: {os.path.join(save_dir, 'run12mnew')}")