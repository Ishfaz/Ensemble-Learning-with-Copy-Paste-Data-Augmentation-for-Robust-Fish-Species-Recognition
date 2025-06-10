# yolo-ensemble modification
ensambling yolo model

### Installation
From the home directory of the project, run the following command to install the required packages:

```bash
pip install -r requirements.txt
```
To train the model on your custom dataset, prepare the dataset that is suiitable for the yolo format. The dataset should be in the following format:
```bash
/dataset
  ├── images
  │   ├── train
  │   ├── val
  │   ├── test  (optional)
  ├── labels
  │   ├── train
  │   ├── val
  │   ├── test  (optional)
  ├── dataset.yaml
```
* The `images/` directory contains all images divided into `train`,`val/`, and optionally `test/` directories.
* The `labels/` directory contains annotation files in YOLO format.
* The `dataset.yaml` file defines class names and paths.

Once you prepare the dataset for training, you can train the YOLO model using various ensembling methods. 

## Ensemble FGE and SWA
For the combined ensemble, you can use the following command to train the model. Run this command from the home directory of the project:

```bash
python Train_SWA.py --data_path your_dataset.yaml --model_path your_custom_model --save_dir checkpoint_directory --epochs 300 --base_epoch 260 -- batch_size 16 --image_size 640 --base_lr 0.0001 --max_lr 0.01
```

The arguments to pass during training are:
* `--data_path` : Path to the dataset.yaml file (**dataset.yaml**). Add your own directory or the code defaults to **coco128.yaml**.
* `--model_path` : Path to the custom model. You can add your custom model. The code defaults to **yolov8l.pt**.
* `--save_dir` : Directory to save the checkpoints. Add your own directory or the code defaults to **fge_checkpoints**.
* `--epochs` : Number of epochs to train the model. The code defaults to **300**.
* `--base_epoch` : The epoch number to start the ensembling and save the base model. The code defaults to **260**

You can pass all these arguments during training to apply the augmentation techniques. The model checkpoints are saved in the directory passed in the argument.

Logs generatd during the training are saved in the **train_logs** directory. It stores the logs from base epoch to the final epoch.

## Fast Geometric Ensembling (FGE)

If you want to train the model using the FGE method, you can use the following command. Run this command from the **fge** direcotry:

```bash
python Train_FGE_8.py --data_path your_dataset.yaml --model_path your_custom_model --save_dir checkpoint_directory --epochs 300 --base_epoch 260 --batch_size  --image_size 640 --base_lr 0.0001 --max_lr 0.01
```
The arguments to pass duiing training are:
* `--data_path` : Path to the dataset.yaml file (**dataset.yaml**). Add your own directory or the code defaults to **coco128.yaml**.
* `--model_path` : Path to the custom model. You can add your custom model. The code defaults to **yolov8m.pt**.
* `--save_dir` : Directory to save the checkpoints. Add your own directory or the code defaults to **fge_checkpoints**.
* `--epochs` : Number of epochs to train the model. The code defaults to **300**.
* `--base_epoch` : The epoch number to start the ensembling and save the base model. The code defaults to **260**.
* `--batch_size` : Batch size for training. The code defaults to **32**.
* `--image_size` : Image size for training. The code defaults to **640**.
* `--base_lr` : The lower learning rate for training. The code defaults to **0.0001**.
* `--max_lr` : The base higher learning rate for training. The code defaults to **0.01**.

The model checkpoints are saved in the directory passed in the argument.

### Testing the FGE model  and SWA on the test dataset

If you want to test the model using the FGE method and SWA, you can use the following code
**Eval_ensemble.py**
**images_folder** = "your image path"
**label_folder** = "your label path"
**output_folder** = "saved results"

##Tracking and Counting with the Ensemble Model

## To perform object tracking and counting using an ensemble model, update paths in Ensemble_track.py
* `--MODEL_DIR`:= '/path/to/ensemble/model/checkpoints'
* `--INPUT_VIDEO_PATH`:= '/path/to/input/video.mov'
* `--OUTPUT_DIR`: = '/path/to/output/results'

