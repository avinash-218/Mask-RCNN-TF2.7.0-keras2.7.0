import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, Callback
import wandb
from wandb.keras import WandbCallback
import json
import copy
from tqdm import tqdm
import warnings
from PIL import Image, ImageDraw
warnings.filterwarnings("ignore")

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class CustomConfig(Config):
    NAME = 'furnitures'  # Override in sub-classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 16 # Override in sub-classes


class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
                # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        coco_json = json.load(open(dataset_dir + "/via_region_data.json"))

        # Add the class names using the base method from utils.Dataset
        source_name = "object"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
                return
            
            self.add_class(source_name, class_id, class_name)
        
        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        
        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.abspath(os.path.join(dataset_dir, image_file_name))
                image_annotations = annotations[image_id]
                
                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def evaluate_model(dataset, model, cfg, name):
    APs = []
    ARs = []
    IOUs = []

    disp_cnt = 0

    for image_id in tqdm(dataset.image_ids):
        # Load image, bounding boxes, and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id)

        yhat = model.detect([image], verbose=0)

        # Extract results for the current image
        r = yhat[0]

        # Calculate statistics, including AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        AR, _ = utils.compute_recall(r["rois"], gt_bbox, iou=0.5)
        APs.append(AP)
        ARs.append(AR)

        max_iou_per_box = np.max(overlaps, axis=1)
        average_iou = np.mean(max_iou_per_box)
        IOUs.append(average_iou)
        
        disp_cnt+=1

        if(disp_cnt % 20 == 0):
            mAP = np.mean(APs)
            mAR = np.mean(ARs)
            F1_score = (2 * mAP * mAR)/(mAP + mAR)
            iou_score = np.mean(IOUs)
            wandb.log({name+"_mAP":mAP, name+"_mAR":mAR, name+"_F1_Score":F1_score, name+"_IOU":iou_score})
            print(f"{disp_cnt}: {name}_mAP = {mAP:.4f}, {name}_mAR = {mAR:.4f}, {name}_F1_Score = {F1_score:.4f}, {name}_IOU = {iou_score:.4f}")

    # Calculate the mean values
    mAP = np.mean(APs)
    mAR = np.mean(ARs)
    F1_score = (2 * mAP * mAR)/(mAP + mAR)
    iou_score = np.mean(IOUs)

    return mAP, mAR, F1_score, iou_score


class CustomWandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)  # Log the metrics to W&B

if __name__ == '__main__':
    # Read the JSON configuration file
    with open('/kaggle/input/maskrcnn-models/API.json', 'r') as config_file:
        config = json.load(config_file)

    # # Retrieve the Wandb API key
    wandb_api_key = config['wandb_api_key']
    wandb.login(key=wandb_api_key)

    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--model_path', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/model/",
                        help='saved model path')
    args = parser.parse_args()

    class InferenceConfig(CustomConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    eval_config = InferenceConfig()
    eval_config.display()

    myrun = wandb.init(
                        project='Eval MaskRCNN',#project name
                        group='Iter3',#set group name
                        name='Run4.1',#set run name
                        resume=False#resume run
                        )

    # # Training dataset.
    # dataset_train = CustomDataset()
    # dataset_train.load_custom(args.dataset, "train")
    # dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # Test dataset
    dataset_test = CustomDataset()
    dataset_test.load_custom(args.dataset, "test")
    dataset_test.prepare()

    # define the model
    model = modellib.MaskRCNN(mode="inference", model_dir='./logs/', config=eval_config)

    # load model weights
    model.load_weights(args.model_path, by_name=True)

    # evaluate model on training dataset
    # print('Evaluating on Train Dataset')
    # train_mAP, train_mAR, train_f1_score, train_iou = evaluate_model(dataset_train, model, eval_config, "Train")
    # wandb.log({"Train_mAP":train_mAP, "Train_mAR":train_mAR, "Train_F1_Score":train_f1_score, "Train_IOU": train_iou})
    # print(f"Train - mAP: {train_mAP:.4f}, mAR: {train_mAR:.4f}, F1: {train_f1_score:.4f}, IOU: {train_iou:.4f}")

    # evaluate model on val dataset
    print('Evaluating on Validation Dataset')
    val_mAP, val_mAR, val_f1_score, val_iou = evaluate_model(dataset_val, model, eval_config, "Val")
    wandb.log({"Val_mAP":val_mAP, "Val_mAR":val_mAR, "Val_F1_Score":val_f1_score, "Val_IOU": val_iou})
    print(f"Validation - mAP: {val_mAP:.4f}, mAR: {val_mAR:.4f}, F1: {val_f1_score:.4f}, IOU: {val_iou:.4f}")

    # evaluate model on test dataset
    print('Evaluating on Test Dataset')
    test_mAP, test_mAR, test_f1_score, test_iou = evaluate_model(dataset_test, model, eval_config, "Test")
    wandb.log({"Test_mAP":test_mAP, "Test_mAR":test_mAR, "Test_F1_Score":test_f1_score, "Test_IOU": test_iou})
    print(f"Test - mAP: {test_mAP:.4f}, mAR: {test_mAR:.4f}, F1: {test_f1_score:.4f}, IOU: {test_iou:.4f}")
    
    wandb.finish()

# python eval.py --dataset=../dataset2/ --model_path=/kaggle/input/furniture/iter2run4.h5
# python eval.py --dataset=/kaggle/input/sfpi-2/dataset2 --model_path=/kaggle/input/furniture/iter2run4.h5