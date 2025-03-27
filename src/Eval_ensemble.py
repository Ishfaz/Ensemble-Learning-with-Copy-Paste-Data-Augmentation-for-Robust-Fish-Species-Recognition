import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from collections import defaultdict
import matplotlib.pyplot as plt
import time

# -------------------------
# Model loading and prediction functions
# -------------------------
def load_models(model_paths):
    """
    Load a list of YOLO models from the given checkpoint paths.
    """
    models_list = []
    for path in model_paths:
        model = YOLO(path)  # load model from checkpoint
        models_list.append(model)
    return models_list

def get_predictions(models, image, conf_thresh=0.25):
    """
    Run inference with each model and return lists of boxes, scores, and labels (normalized).
    """
    boxes_list = []
    scores_list = []
    labels_list = []
    h, w = image.shape[:2]
    for model in models:
        results = model(image)
        if len(results) == 0:
            continue
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # absolute coordinates
        scores = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        keep = scores >= conf_thresh
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= w
        boxes_norm[:, [1, 3]] /= h
        boxes_list.append(boxes_norm.tolist())
        scores_list.append(scores.tolist())
        labels_list.append(labels.tolist())
    return boxes_list, scores_list, labels_list

def ensemble_predictions(boxes_list, scores_list, labels_list, iou_thr=0.55, skip_box_thr=0.0, conf_out=0.25):
    """
    Combine predictions from multiple models using Weighted Boxes Fusion (WBF).
    """
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='avg'
    )
    if len(scores) > 0:
        filtered = [(box, score, label) for box, score, label in zip(boxes, scores, labels) if score >= conf_out]
        if filtered:
            boxes, scores, labels = zip(*filtered)
            boxes = np.array(boxes)
            scores = np.array(scores)
            labels = np.array(labels)
        else:
            boxes, scores, labels = np.empty((0, 4)), np.array([]), np.array([])
    else:
        boxes, scores, labels = np.empty((0, 4)), np.array([]), np.array([])
    return boxes, scores, labels

def denormalize_boxes(boxes, image_shape):
    """
    Convert normalized box coordinates back to absolute pixel values.
    """
    h, w = image_shape[:2]
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, axis=0)
    boxes_abs = boxes.copy()
    boxes_abs[:, [0, 2]] *= w
    boxes_abs[:, [1, 3]] *= h
    return boxes_abs

# -------------------------
# Ground truth functions for mAP50
# -------------------------
def load_ground_truth(label_path, image_shape):
    """
    Load ground-truth boxes from a label file (YOLO format) and convert to absolute xyxy.
    
    Args:
        label_path (str): Path to the label text file.
        image_shape (tuple): (height, width, channels) of the corresponding image.
    
    Returns:
        np.ndarray: Array of ground truth boxes in xyxy absolute coordinates.
        np.ndarray: Array of ground truth class labels.
    """
    h, w = image_shape[:2]
    gt_boxes = []
    gt_labels = []
    if not os.path.exists(label_path):
        return np.empty((0, 4)), np.array([])
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, bw, bh = map(float, parts)
            # Convert YOLO format (normalized center, width, height) to xyxy absolute
            x1 = (x_center - bw / 2) * w
            y1 = (y_center - bh / 2) * h
            x2 = (x_center + bw / 2) * w
            y2 = (y_center + bh / 2) * h
            gt_boxes.append([x1, y1, x2, y2])
            gt_labels.append(int(cls))
    return np.array(gt_boxes), np.array(gt_labels)

def compute_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compute_ap(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """
    Compute average precision for a single image using a simplified algorithm.
    
    Args:
        pred_boxes (np.ndarray): Predicted boxes in absolute coordinates.
        pred_scores (np.ndarray): Confidence scores for predicted boxes.
        pred_labels (np.ndarray): Class labels for predicted boxes.
        gt_boxes (np.ndarray): Ground truth boxes in absolute coordinates.
        gt_labels (np.ndarray): Ground truth class labels.
        iou_threshold (float): IoU threshold to consider a prediction as true positive.
    
    Returns:
        float: Average precision for the image.
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0

    # Sort predictions by descending confidence
    order = pred_scores.argsort()[::-1]
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    
    matched = np.zeros(len(gt_boxes), dtype=bool)
    tp = 0
    for box, label in zip(pred_boxes, pred_labels):
        # For each predicted box, find the best matching gt box of the same class
        best_iou = 0
        best_idx = -1
        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_label != label:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_threshold and best_idx != -1 and not matched[best_idx]:
            tp += 1
            matched[best_idx] = True
    precision = tp / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    return precision

def compute_precision_recall(y_true, probas_pred):
    """Simple implementation of precision-recall curve calculation"""
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    
    for threshold in thresholds:
        y_pred = (probas_pred >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls)

def evaluate_detections(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    """Compute detailed metrics for object detection"""
    metrics = {
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0,
        'iou_scores': [],
        'precisions': [],
        'recalls': [],
        'ap': 0.0
    }
    
    if len(gt_boxes) == 0:
        if len(pred_boxes) == 0:
            return metrics
        metrics['false_positives'] = len(pred_boxes)
        return metrics

    # Sort by confidence
    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_labels = pred_labels[order]

    matched = np.zeros(len(gt_boxes), dtype=bool)
    
    for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        best_iou = 0
        best_gt_idx = -1
        
        for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_label != label or matched[j]:
                continue
            iou = compute_iou(box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched[best_gt_idx] = True
            metrics['true_positives'] += 1
            metrics['iou_scores'].append(best_iou)
        else:
            metrics['false_positives'] += 1
            
        # Calculate precision and recall at each detection
        tp = np.sum(matched)
        fp = i + 1 - tp
        recall = tp / len(gt_boxes)
        precision = tp / (tp + fp)
        metrics['precisions'].append(precision)
        metrics['recalls'].append(recall)
    
    metrics['false_negatives'] = len(gt_boxes) - np.sum(matched)
    if metrics['precisions']:
        precisions = np.array(metrics['precisions'])
        recalls = np.array(metrics['recalls'])
        # Calculate AP as area under PR curve using simple interpolation
        metrics['ap'] = np.mean(precisions)
    
    return metrics

def plot_metrics(metrics_dict, output_folder):
    """Plot and save detection metrics"""
    plt.figure(figsize=(15, 5))
    
    # Precision-Recall curve
    plt.subplot(131)
    plt.plot(metrics_dict['recalls'], metrics_dict['precisions'])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\nAP50: {metrics_dict["map50"]:.4f}')
    
    # IoU distribution
    plt.subplot(132)
    plt.hist(metrics_dict['iou_scores'], bins=20, range=(0, 1))
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title(f'IoU Distribution\nMean IoU: {np.mean(metrics_dict["iou_scores"]):.4f}')
    
    # Confidence distribution
    plt.subplot(133)
    plt.hist(metrics_dict['confidence_scores'], bins=20, range=(0, 1))
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'ensemble_metrics.png'))
    plt.close()

def calculate_f1_score(precision, recall):
    """Calculate F1 score from precision and recall"""
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def save_summary(output_folder, stats, fps_stats):
    """Save detection summary with FPS information to a file"""
    summary_path = os.path.join(output_folder, 'ensemble_summary.txt')
    
    # Calculate metrics
    precision = stats['true_positives'] / (stats['true_positives'] + stats['false_positives']) if stats['true_positives'] + stats['false_positives'] > 0 else 0
    recall = stats['true_positives'] / (stats['true_positives'] + stats['false_negatives']) if stats['true_positives'] + stats['false_negatives'] > 0 else 0
    f1_score = calculate_f1_score(precision, recall)
    
    with open(summary_path, 'w') as f:
        f.write("Ensemble Detection Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Performance Metrics:\n")
        f.write(f"Average FPS: {fps_stats['avg_fps']:.2f}\n")
        f.write(f"Average time per frame: {fps_stats['avg_time_ms']:.1f}ms\n")
        f.write(f"Total processing time: {fps_stats['total_time']:.2f}s\n\n")
        
        f.write(f"Total images processed: {stats['total_images']}\n")
        f.write(f"Images with detections: {stats['images_with_detections']}\n")
        f.write(f"Detection rate: {stats['images_with_detections']/stats['total_images']*100:.2f}%\n\n")
        
        f.write("Detection Statistics:\n")
        f.write(f"Total ground truth boxes: {stats['total_gt_boxes']}\n")
        f.write(f"Total detections: {stats['total_detections']}\n")
        f.write(f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}\n\n")
        
        f.write("Quality Metrics:\n")
        f.write(f"Mean IoU: {np.mean(stats['all_ious']) if stats['all_ious'] else 0:.4f}\n")
        f.write(f"Mean Confidence: {np.mean(stats['all_scores']) if stats['all_scores'] else 0:.4f}\n\n")
        
        f.write("Classification Metrics:\n")
        f.write(f"True Positives: {stats['true_positives']}\n")
        f.write(f"False Positives: {stats['false_positives']}\n")
        f.write(f"False Negatives: {stats['false_negatives']}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1_score:.4f}\n\n")
        
        f.write(f"Results saved to: {output_folder}\n")
        f.write("=" * 50 + "\n")
    
    return summary_path

def process_images(image_folder, label_folder, output_folder, models_list, **kwargs):
    """Process images with ensemble models and compute detailed metrics"""
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize counters
    total_images = 0
    images_with_detections = 0
    total_detections = 0
    total_gt_boxes = 0
    all_ious = []
    all_scores = []
    
    # Add counters for classification metrics
    stats = {
        'total_images': 0,
        'images_with_detections': 0,
        'total_detections': 0,
        'total_gt_boxes': 0,
        'all_ious': [],
        'all_scores': [],
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    # Process each image
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    total_time = 0
    total_frames = 0
    
    # FPS tracking
    fps_stats = {
        'total_time': 0,
        'total_frames': 0,
        'per_image_fps': [],
        'avg_fps': 0,
        'avg_time_ms': 0
    }
    
    print("\nProcessing images:")
    print("="*50)
    
    for image_file in image_files:
        start_time = time.time()
        
        stats['total_images'] += 1
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get ground truth boxes
        base_name = os.path.splitext(image_file)[0]
        gt_boxes, gt_labels = load_ground_truth(os.path.join(label_folder, f"{base_name}.txt"), image.shape)
        stats['total_gt_boxes'] += len(gt_boxes)
        
        # Get ensemble predictions
        boxes_list, scores_list, labels_list = get_predictions(models_list, image_rgb, conf_thresh=kwargs.get('conf_thresh', 0.25))
        ensemble_boxes, ensemble_scores, ensemble_labels = ensemble_predictions(
            boxes_list, scores_list, labels_list,
            iou_thr=kwargs.get('iou_thr', 0.55),
            skip_box_thr=kwargs.get('skip_box_thr', 0.0),
            conf_out=kwargs.get('conf_out', 0.25)
        )
        
        # Track detections
        if len(ensemble_boxes) > 0:
            stats['images_with_detections'] += 1
            stats['total_detections'] += len(ensemble_boxes)
        
        # Convert normalized boxes to absolute coordinates
        ensemble_boxes_abs = denormalize_boxes(ensemble_boxes, image.shape)
        
        # Compute metrics
        metrics = evaluate_detections(
            ensemble_boxes_abs, ensemble_scores, ensemble_labels,
            gt_boxes, gt_labels, iou_threshold=kwargs.get('iou_eval', 0.5)
        )
        
        # Store metrics
        if metrics['iou_scores']:
            stats['all_ious'].extend(metrics['iou_scores'])
        if len(ensemble_scores) > 0:
            stats['all_scores'].extend(ensemble_scores)
            
        # Update classification metrics
        stats['true_positives'] += metrics['true_positives']
        stats['false_positives'] += metrics['false_positives']
        stats['false_negatives'] += metrics['false_negatives']
        
        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time
        total_frames += 1
        
        # Track FPS
        fps_stats['total_time'] += inference_time
        fps_stats['total_frames'] += 1
        fps = 1.0 / inference_time
        fps_stats['per_image_fps'].append(fps)
        
        # Print per-image FPS
        fps = 1.0 / inference_time
        print(f"Image: {image_file}")
        print(f"Processing time: {inference_time:.3f}s")
        print(f"FPS: {fps:.2f}")
        print("-"*50)
        
        # Print per-image results
        print(f"Image: {image_file}")
        print(f"GT boxes: {len(gt_boxes)}, Predictions: {len(ensemble_boxes)}")
        print(f"IoUs: {np.mean(metrics['iou_scores']) if metrics['iou_scores'] else 0:.4f}")
        print("-"*50)
        
        # Visualize and save
        vis_img = image.copy()
        # Draw ground truth boxes in red
        for box in gt_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for ground truth
            
        # Draw predictions in green
        for box, score in zip(ensemble_boxes_abs, ensemble_scores):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for predictions
            cv2.putText(vis_img, f"{score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(output_folder, f"pred_{image_file}"), vis_img)

    # Calculate average FPS
    avg_fps = total_frames / total_time
    
    # Calculate final FPS statistics
    fps_stats['avg_fps'] = fps_stats['total_frames'] / fps_stats['total_time']
    fps_stats['avg_time_ms'] = (fps_stats['total_time'] / fps_stats['total_frames']) * 1000
    
    # Save summary with FPS info
    summary_path = save_summary(output_folder, stats, fps_stats)
    
    # Print performance summary
    print("\nPerformance Metrics:")
    print(f"Average FPS: {fps_stats['avg_fps']:.2f}")
    print(f"Average time per frame: {fps_stats['avg_time_ms']:.1f}ms")
    print(f"Total processing time: {fps_stats['total_time']:.2f}s")
    print(f"\nDetailed summary saved to: {summary_path}")

    # Print detailed summary
    print("\nEnsemble Detection Summary")
    print("="*50)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Detection rate: {stats['images_with_detections']/stats['total_images']*100:.2f}%")
    print(f"Total ground truth boxes: {stats['total_gt_boxes']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per image: {stats['total_detections']/stats['total_images']:.2f}")
    print(f"Mean IoU (of successful detections): {np.mean(stats['all_ious']) if stats['all_ious'] else 0:.4f}")
    print(f"Mean Confidence: {np.mean(stats['all_scores']) if stats['all_scores'] else 0:.4f}")
    print(f"Results saved to: {output_folder}")
    print("="*50)
    
    # Save detailed summary
    summary_path = save_summary(output_folder, stats, fps_stats)
    print(f"\nDetailed summary saved to: {summary_path}")

if __name__ == "__main__":
    # Define weights directory
    weights_dir = "/cluster/home/ishfaqab/Saithes_prepared/Y8_12"
    
    # Get all .pt model files from the weights directory
    model_paths = sorted([
        os.path.join(weights_dir, f) 
        for f in os.listdir(weights_dir) 
        if f.endswith('.pt') 
    ])
    
    if not model_paths:
        raise ValueError(f"No model files found in {weights_dir}")

    print(f"Found {len(model_paths)} model files")
    models_list = load_models(model_paths)
    
    # Define folders
    images_folder = "/cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/valid/images"
    label_folder = "/cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/valid/labels"
    output_folder = "/cluster/home/ishfaqab/Fish_annotation_NTNU/FGE/run3_ensamble_8-12"

    # Process all images, compute mAP50, and save output images
    process_images(images_folder, label_folder, output_folder, models_list,
                   conf_thresh=0.25, iou_thr=0.5, skip_box_thr=0.0, conf_out=0.25, iou_eval=0.5)
