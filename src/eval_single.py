import os
from Eval_ensemble import process_images, load_models

def evaluate_single_model(model_path, model_name):
    """Evaluate a single model and save results"""
    print(f"\nEvaluating {model_name}...")
    print("="*50)
    
    # Create output directory for this model
    output_folder = f"/cluster/home/ishfaqab/Fish_annotation_NTNU/Single_Model_Results_different_data/{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    
    # Load single model
    model = load_models([model_path])
    
    # Define data folders
    images_folder = "/cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/valid/images"
    label_folder = "/cluster/home/ishfaqab/Saithes_prepared/dataIfiltered/valid/labels"
    
    # Process images with single model
    process_images(
        images_folder, 
        label_folder, 
        output_folder, 
        model,
        conf_thresh=0.25, 
        iou_thr=0.50, 
        skip_box_thr=0.0, 
        conf_out=0.25, 
        iou_eval=0.5
    )

def main():
    # Model configurations
    models = {
        # 'YOLO12l': {
        #     'path': "/cluster/home/ishfaqab/Saithes_prepared/results/12m_filtered_data/runL/weights/best.pt",
        #     'description': "YOLO12l Model"
        # },
        # 'YOLOv8l': {
        #     'path': "/cluster/home/ishfaqab/Saithes_prepared/results/8m_L_filtered_data/runL/weights/best.pt",
        #     'description': "YOLO8 Model"
        # },
        'yolo8m': {
            'path': "/cluster/home/ishfaqab/Saithes_prepared/Y8_12/best_8.pt",
            'description': "SWA Model"
        }
    }
    
    # Evaluate each model separatel
    for model_name, model_info in models.items():
        evaluate_single_model(model_info['path'], model_name)
        print(f"Completed evaluation of {model_info['description']}\n")

if __name__ == "__main__":
    main()
