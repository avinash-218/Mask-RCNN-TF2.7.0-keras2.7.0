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
        # Add classes. We have 16 classes to add.
        self.add_class("object", 1, "armchair")
        self.add_class("object", 2, "bed")
        self.add_class("object", 3, "door1")
        self.add_class("object", 4, "door2")
        self.add_class("object", 5, "sink1")
        self.add_class("object", 6, "sink2")
        self.add_class("object", 7, "sink3")
        self.add_class("object", 8, "sink4")
        self.add_class("object", 9, "sofa1")
        self.add_class("object", 10, "sofa2")
        self.add_class("object", 11, "table1")
        self.add_class("object", 12, "table2")
        self.add_class("object", 13, "table3")
        self.add_class("object", 14, "tub")
        self.add_class("object", 15, "window1")
        self.add_class("object", 16, "window2")

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        data = json.load(open(dataset_dir + "/via_region_data.json"))

        images = data['images']
        annotations = data['annotations']

        annotation_cnt = 0
        len_annotations = len(annotations)

        for i in images:
            image_id = i['file_name']
            image_path = dataset_dir + '/' +image_id
            height = i['height']
            width = i['width']
            identity = i['id'] #image id

            j = annotation_cnt
            num_ids = []
            polygons = []

            while j<len_annotations: #extract data from all annotations of ith image
                if(identity == annotations[j]['image_id']): #same image
                    num_ids.append(annotations[j]['category_id']) # append category ids in num_ids list

                    segmentation = annotations[j]['segmentation']
                    all_points_x = segmentation[0][::2]
                    all_points_y = segmentation[0][1::2]
                    polygon = {'name':'polygon', 'all_points_x':all_points_x, 'all_points_y':all_points_y}

                    polygons.append(polygon)
                    j+=1

                else: #different image
                    annotation_cnt = j
                    break

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=image_id,  # use file name as a unique image id
                path=image_path,
                width=width,
                height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bottle dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
            
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def evaluate_model(dataset, model, cfg, iou_threshold=0.5):
    APs = []
    ARs = []
    F1_scores = []
    IOUs =[]
    dices = []

    disp_cnt = 0

    for image_id in tqdm(dataset.image_ids):
        # Load image, bounding boxes, and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id)

        # Make prediction - detect returns the following:
        # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        # class_ids: [N] int class IDs
        # scores: [N] float probability scores for the class IDs
        # masks: [H, W, N] instance binary masks
        yhat = model.detect([image], verbose=0)

        # Extract results for the current image
        r = yhat[0]

        # Calculate statistics, including AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'], iou_threshold=iou_threshold)
        AR, _ = utils.compute_recall(r["rois"], gt_bbox, iou=0.5)
        APs.append(AP)
        ARs.append(AR)
        F1_scores.append((2* (np.mean(precisions) * np.mean(recalls)))/(np.mean(precisions) + np.mean(recalls)))
        
        max_iou_per_box = np.max(overlaps, axis=1)
        average_iou = np.mean(max_iou_per_box)
        dice = 2*average_iou / (average_iou + 1)

        # Store individual values
        IOUs.append(average_iou)
        dices.append(dice)
        disp_cnt+=1

        if(disp_cnt % 10 == 0):
            print(np.mean(APs), np.mean(ARs), ( (2*np.mean(APs)*np.mean(ARs)) / (np.mean(APs) + np.mean(ARs))))

    # Calculate the mean values
    mAP = mean(APs)
    mAR = mean(ARs)
    F1_score = (2 * mAP * mAR)/(mAP + mAR)
    iou = np.mean(IOUs)
    dice = np.mean(dices)

    return mAP, mAR, F1_score, iou, dice


class CustomWandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)  # Log the metrics to W&B

if __name__ == '__main__':
    # Read the JSON configuration file
    with open('/kaggle/input/furniture/API.json', 'r') as config_file:
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
                        group='Iter1',#set group name
                        name='Run1',#set run name
                        resume=False#resume run
                        )

    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

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
    # train_mAP, train_mAR, train_f1_score, train_iou, train_dice = evaluate_model(dataset_train, model, eval_config)
    # print(f"Train - mAP: {train_mAP:.4f}, mAR: {train_mAR:.4f}, F1: {train_f1_score:.4f}, IOU: {train_iou:.4f}, Dice: {train_dice:.4f}")

    # evaluate model on val dataset
    print('Evaluating on Validation Dataset')
    val_mAP, val_mAR, val_f1_score, val_iou, val_dice = evaluate_model(dataset_val, model, eval_config)
    print(f"Validation - mAP: {val_mAP:.4f}, mAR: {val_mAR:.4f}, F1: {val_f1_score:.4f}, IOU: {val_iou:.4f}, Dice: {val_dice:.4f}")

    # evaluate model on test dataset
    print('Evaluating on Test Dataset')
    test_mAP, test_mAR, test_f1_score, test_iou, test_dice = evaluate_model(dataset_test, model, eval_config)
    print(f"Test - mAP: {test_mAP:.4f}, mAR: {test_mAR:.4f}, F1: {test_f1_score:.4f}, IOU: {test_iou:.4f}, Dice: {test_dice:.4f}")

    
    wandb.log({"Val mAP": val_mAP, "Test mAP": test_mAP,
            "Val mAR": val_mAR, "Test mAR": test_mAR,
            "Val F1": val_f1_score, "Test mean F1": test_f1_score,
            "Val IOU": val_iou, "Test IOU": test_iou,
            "Val Dice": val_dice, "Test Dice": test_dice})


    wandb.log({"Train mAP": train_mAP, "Val mAP": val_mAP, "Test mAP": test_mAP,
            "Train mAR": train_mAR, "Val mAR": val_mAR, "Test mAR": test_mAR,
            "Train F1": train_f1_score, "Val F1": val_f1_score, "Test mean F1": test_f1_score,
            "Train IOU": train_iou, "Val IOU": val_iou, "Test IOU": test_iou,
            "Train Dice": train_dice, "Val Dice": val_dice, "Test Dice": test_dice})

    wandb.finish()

# python eval.py --dataset=../dataset/ --logs=./logs