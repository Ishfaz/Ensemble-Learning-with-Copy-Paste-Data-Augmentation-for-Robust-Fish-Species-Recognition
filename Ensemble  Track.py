
from ultralytics import YOLO
import os
import time
import cv2
import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion

# Configuration
MODEL_DIR = '/cluster/home/ishfaqab/Saithes_prepared/New_Folder/new_data/results/Y_8_12'  # model directory
INPUT_VIDEO_PATH = "/cluster/home/ishfaqab/Fish_annotation_NTNU/track/IMG_7941.mov"# input the vedio path
OUTPUT_DIR = "/cluster/home/ishfaqab/Fish_annotation_NTNU/track/results"
CONF_THRESH = 0.35  # Slightly higher confidence threshold
IOU_THRESH = 0.6    # Higher IoU threshold for better NMS
TRACK_IOU_THRESH = 0.6  # IoU threshold for tracking
SAITHE_CLASS = 2  # Saithe is class 2
POLLOCK_CLASS = 1  # Pollock is class 1
TARGET_CLASSES = [POLLOCK_CLASS, SAITHE_CLASS]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Extract filename
filename = os.path.basename(INPUT_VIDEO_PATH)
filename_without_ext = os.path.splitext(filename)[0]
output_path = os.path.join(OUTPUT_DIR, f"{filename_without_ext}_ensemble_tracked.mp4")

print(f"Processing file: {filename}")
print(f"Using ensemble detection for Saithe (class {SAITHE_CLASS}) and Pollock (class {POLLOCK_CLASS})...")

# Load all models in the directory
print("\nLoading YOLO models for ensemble...")
models = []
class_names = {}

model_files = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
for model_path in sorted(model_files):
    try:
        model = YOLO(model_path)
        models.append(model)
        print(f"  - Loaded: {os.path.basename(model_path)}")
        
        # Store class names from the first model
        if not class_names:
            class_names = model.names
    except Exception as e:
        print(f"  - Error loading {model_path}: {str(e)}")

if not models:
    raise ValueError("No models were successfully loaded!")

print(f"Successfully loaded {len(models)} models")

# Get video properties
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Set weights for ensemble (equal weights)
weights = [1.0 / len(models)] * len(models)

def get_ensemble_predictions(frame):
    """
    Get ensemble predictions from all models using WBF
    
    Args:
        frame: The input image/frame
        
    Returns:
        Boxes, scores, and labels from ensemble predictions
    """
    all_predictions = []
    
    # Get predictions from each model
    for model in models:
        results = model(frame, verbose=False, conf=CONF_THRESH, classes=TARGET_CLASSES)[0]
        
        if results.boxes is None or len(results.boxes) == 0:
            # No detections from this model
            all_predictions.append({
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            })
            continue
        
        # Convert to numpy arrays
        boxes = results.boxes.xyxy.cpu().numpy()
        scores = results.boxes.conf.cpu().numpy()
        labels = results.boxes.cls.cpu().numpy().astype(int)
        
        # Normalize boxes for WBF (expects [0-1] coordinates)
        norm_boxes = boxes.copy()
        norm_boxes[:, 0] /= width
        norm_boxes[:, 1] /= height
        norm_boxes[:, 2] /= width
        norm_boxes[:, 3] /= height
        
        all_predictions.append({
            'boxes': norm_boxes,
            'scores': scores,
            'labels': labels
        })
    
    # Apply WBF if we have any detections
    if any(len(p['boxes']) > 0 for p in all_predictions):
        boxes_list = [p['boxes'] for p in all_predictions]
        scores_list = [p['scores'] for p in all_predictions]
        labels_list = [p['labels'] for p in all_predictions]
        
        # Apply WBF
        wbf_boxes, wbf_scores, wbf_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=IOU_THRESH,
            skip_box_thr=CONF_THRESH
        )
        
        # Convert normalized boxes back to absolute coordinates
        if len(wbf_boxes) > 0:
            wbf_boxes[:, 0] *= width
            wbf_boxes[:, 1] *= height
            wbf_boxes[:, 2] *= width
            wbf_boxes[:, 3] *= height
        
        return wbf_boxes, wbf_scores, wbf_labels
    else:
        return np.array([]), np.array([]), np.array([])

def tracking_yolov8_style():
    """
    Run tracking using YOLOv8 style approach with ensemble predictions
    This uses a direct approach similar to how YOLO models use BotSORT
    """
    # Set up output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create tracker model for BotSORT
    print("\nInitializing YOLOv8+BotSORT Tracker...")
    # Use the first model for tracking framework (we'll replace its detections)
    tracker_model = models[0]
    
    # Open input video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {INPUT_VIDEO_PATH}")
    
    # Tracker gets detections from a base model (we provide our ensemble detections to it)
    results = None  # Placeholder for tracking results
    
    start_time = time.time()
    frame_idx = 0
    
    # For counting fish over the whole video
    all_saithe_ids = set()  # Track all unique Saithe IDs seen throughout video
    all_pollock_ids = set()  # Track all unique Pollock IDs seen throughout video
    
    # For counting fish in the current frame
    current_saithe_count = 0
    current_pollock_count = 0
    
    # Process each frame
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get ensemble predictions for this frame
        ensemble_boxes, ensemble_scores, ensemble_labels = get_ensemble_predictions(frame)
        
        # Process with tracking
        if frame_idx == 0:
            # First frame - initialize tracker with our ensemble detections
            results = tracker_model.track(
                source=frame,
                persist=True,
                tracker="botsort.yaml",
                conf=CONF_THRESH,
                classes=TARGET_CLASSES,
                iou=TRACK_IOU_THRESH,
                verbose=False
            )[0]
        else:
            # Subsequent frames - use same tracker for persistence
            results = tracker_model.track(
                source=frame,
                persist=True,
                tracker="botsort.yaml",
                conf=CONF_THRESH,
                classes=TARGET_CLASSES,
                iou=TRACK_IOU_THRESH,
                verbose=False
            )[0]
        
        # Start with a clean frame for visualization
        frame_with_boxes = frame.copy()
        
        # Reset current frame counts
        current_saithe_count = 0
        current_pollock_count = 0
        
        # Update fish counts based on tracking results
        if hasattr(results.boxes, 'id') and results.boxes.id is not None:
            # Get tracking details
            track_ids = results.boxes.id.cpu().numpy()
            track_boxes = results.boxes.xyxy.cpu().numpy()
            track_labels = results.boxes.cls.cpu().numpy()
            track_scores = results.boxes.conf.cpu().numpy()
            
            # Count fish in current frame and add to unique IDs
            for label, track_id in zip(track_labels, track_ids):
                if label == SAITHE_CLASS:
                    all_saithe_ids.add(int(track_id))  # Add to all-time unique IDs
                    current_saithe_count += 1  # Count in current frame
                elif label == POLLOCK_CLASS:
                    all_pollock_ids.add(int(track_id))  # Add to all-time unique IDs
                    current_pollock_count += 1  # Count in current frame
            
            # Draw tracking boxes and IDs
            for i, (box, score, label, track_id) in enumerate(zip(track_boxes, track_scores, track_labels, track_ids)):
                x1, y1, x2, y2 = map(int, box)
                track_id = int(track_id)
                label = int(label)
                
                # Get class name
                class_name = class_names.get(label, f"Class {label}")
                
                # Set colors based on class - matching your example images
                if label == SAITHE_CLASS:
                    box_color = (255, 255, 255)  # White box for Saithe
                    label_bg_color = (0, 0, 255)  # Red background for Saithe (appears as navy blue in images)
                elif label == POLLOCK_CLASS:
                    box_color = (0, 255, 255)  # Yellow/cyan box for Pollock
                    label_bg_color = (255, 0, 0)  # Blue background for Pollock
                else:
                    box_color = (255, 255, 255)  # White for others
                    label_bg_color = (0, 165, 255)  # Default background
                
                # Draw bounding box
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                
                # Create label in format "id:3 Pollock 0.88" with larger text
                label_text = f"id:{track_id} {class_name} {score:.2f}"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                
                # Background rectangle for text
                cv2.rectangle(frame_with_boxes, 
                            (x1, y1 - text_size[1] - 10), 
                            (x1 + text_size[0] + 10, y1), 
                            label_bg_color, -1)  # Background color based on class
                
                # Text in white
                cv2.putText(frame_with_boxes, label_text, 
                          (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Calculate total unique fish
        total_unique_fish = len(all_saithe_ids) + len(all_pollock_ids)
        current_total_fish = current_saithe_count + current_pollock_count
        
        # Add fish counts at the top (matching your example)
        # Create a semi-transparent overlay at the top
        overlay = frame_with_boxes[0:60, 0:width].copy()
        cv2.rectangle(frame_with_boxes, (0, 0), (width, 60), (0, 0, 0), -1)  # Black background
        cv2.addWeighted(overlay, 0.2, frame_with_boxes[0:60, 0:width], 0.8, 0, frame_with_boxes[0:60, 0:width])
        
        # Add Saithe count with green color - shows number of Saithe in current frame
        cv2.putText(frame_with_boxes, f"Saithe: {len(all_saithe_ids)}", (20, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)  # Green text
                  
        # Add Pollock count in cyan/turquoise - shows number of Pollock in current frame
        pollock_x = 300  # Position for Pollock count
        cv2.putText(frame_with_boxes, f"Pollock: {len(all_pollock_ids)}", (pollock_x, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)  # Cyan text
        
        # Add total unique fish count on the right
        total_text = f"Total Fish: {total_unique_fish}"
        total_text_size = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.putText(frame_with_boxes, total_text, 
                  (width - total_text_size[0] - 20, 40), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (218, 165, 32), 2, cv2.LINE_AA)  # Gold color
        
        # Add ensemble label (smaller and at top right)
        ensemble_text = "ENSEMBLE+BOTSORT"
        cv2.putText(frame_with_boxes, ensemble_text, (width//2 - 100, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add progress info at the bottom
        progress = f"Frame: {frame_idx+1}/{total_frames} ({(frame_idx+1)/total_frames*100:.1f}%)"
        cv2.putText(frame_with_boxes, progress, (10, height-20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Write frame to output video
        out.write(frame_with_boxes)
        
        # Print progress occasionally
        if frame_idx % 100 == 0 or frame_idx == total_frames-1:
            elapsed_time = time.time() - start_time
            est_total_time = elapsed_time / (frame_idx+1) * total_frames
            remaining_time = max(0, est_total_time - elapsed_time)
            
            print(f"Processed frame {frame_idx+1}/{total_frames} ({(frame_idx+1)/total_frames*100:.1f}%)")
            print(f"Elapsed: {elapsed_time:.2f}s, Estimated remaining: {remaining_time:.2f}s")
            print(f"Current frame - Saithe: {current_saithe_count}, Pollock: {current_pollock_count}, Total: {current_total_fish}")
            print(f"Unique fish so far - Saithe: {len(all_saithe_ids)}, Pollock: {len(all_pollock_ids)}, Total: {total_unique_fish}")
        
        frame_idx += 1
    
    # Clean up
    cap.release()
    out.release()
    
    # Print completion info
    elapsed_time = time.time() - start_time
    fps_processing = total_frames / elapsed_time
    print(f"\nProcessing complete! Total time: {elapsed_time:.2f}s (Avg: {fps_processing:.2f} FPS)")
    print(f"Final unique fish count - Saithe: {len(all_saithe_ids)}, Pollock: {len(all_pollock_ids)}, Total: {total_unique_fish}")
    print(f"Tracked video saved to: {output_path}")

# Main entry point
if __name__ == "__main__":
    try:
        tracking_yolov8_style()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
