import torch
from ultralytics import YOLO
import os
from torch.optim.lr_scheduler import CyclicLR
import torch.optim.swa_utils as swa_utils
import logging
import argparse
import albumentations as A
from ultralytics.utils import LOGGER, colorstr
import numpy as np
from torch.utils.data import DataLoader
from ultralytics.data import YOLODataset

# Set logging level
logging.getLogger("ultralytics").setLevel(logging.WARNING)

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

def main(data_path: str, model_path: str, save_dir: str, epochs: int, base_epoch: int, 
         batch_size: int, image_size: int, base_lr: float, max_lr: float):
    
    os.makedirs(save_dir, exist_ok=True) ## Create save directory if it does not exist
    model = YOLO(model_path) ## Load model
    
    # Add custom augmentation callbacks
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_start", on_train_batch_start)
    
    ### Defining cyclical LR
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.01, 
                                momentum=0.9)
    scheduler = CyclicLR(optimizer, 
                         base_lr=base_lr, 
                         max_lr=max_lr, 
                         step_size_up=3, 
                         mode='triangular')
    
    ### Selecting the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining Model on: {device}\n")
    
    ### Training the model for initial epochs
    model.train(data=data_path, 
                epochs=base_epoch, 
                batch=batch_size, 
                imgsz=image_size, 
                device=device, 
                save_period=10,
                verbose=False)
    
    ### Saving the base model
    model.save(f"{save_dir}/base_model_{base_epoch}.pt")
    print(f"\nBase model saved at {save_dir}/base_model_{base_epoch}.pt\n")
    
    # Initialize SWA model and scheduler
    swa_model = swa_utils.AveragedModel(model.model)
    swa_scheduler = swa_utils.SWALR(optimizer, anneal_strategy="cos", anneal_epochs=5, swa_lr=0.01)
    
    ### Training the model using SWA
    for epoch in range(base_epoch+1, epochs+1):
        print(f"\n\nTraining Epoch: {epoch}")
        model.train(data=data_path, 
                    epochs=1, 
                    batch=batch_size, 
                    imgsz=image_size, 
                    device=device,
                    verbose=False)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        ### Applying SWA update after base_epoch
        if epoch > base_epoch:
            swa_model.update_parameters(model.model)
            swa_scheduler.step()
        
        ### Save the SWA model periodically
        if epoch % 6 == 0:
            # Save the current model state
            swa_model_path = f"{save_dir}/swa_model_{epoch}.pt"
            model.save(swa_model_path)
            print(f"\nSWA model saved at {swa_model_path}\n")
    
    ### Save the final SWA model
    final_swa_path = f"{save_dir}/final_swa_model.pt"
    model.save(final_swa_path)
    print(f"\nFinal SWA model saved at {final_swa_path}\n")
    
    print("\n Training Completed!!!\n")

if __name__ == "__main__":
    ### Taking input arguments from the user
    parser = argparse.ArgumentParser(description="Train YOLOv8 with custom augmentations and SWA")

    parser.add_argument("--data_path", type=str, default="coco128.yaml", help="Path to data.yaml")
    parser.add_argument("--model_path", type=str, default="yolov8m.pt", help="Pretrained Model")
    parser.add_argument("--save_dir", type=str, default="swa_checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--base_epoch", type=int, default=260, help="Base epoch for SWA")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=640, help="Image size for training")
    parser.add_argument("--base_lr", type=float, default=0.0001, help="low learning rate for FGE")
    parser.add_argument("--max_lr", type=float, default=0.01, help="high learning rate for FGE")

    args = parser.parse_args()
    
    main(data_path=args.data_path,
         model_path=args.model_path,
         save_dir=args.save_dir,
         epochs=args.epochs,
         base_epoch=args.base_epoch,
         batch_size=args.batch_size,
         image_size=args.image_size,
         base_lr=args.base_lr,
         max_lr=args.max_lr)

