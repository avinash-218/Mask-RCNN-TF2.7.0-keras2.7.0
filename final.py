"""
Mask R-CNN
Train on the floor plan dataset and implement color splash effect.
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
------------------------------------------------------------
Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:
    # Train a new model starting from pre-trained COCO weights
    python3 bottle.py train --dataset=/home/datascience/Workspace/maskRcnn/Mask_RCNN-master/samples/bottle/dataset --weights=coco
    # Resume training a model that you had trained earlier
    python3 bottle.py train --dataset=/path/to/bottle/dataset --weights=last
    # Train a new model starting from ImageNet weights
    python3 bottle.py train --dataset=/path/to/bottle/dataset --weights=imagenet
    # Apply color splash to an image
    python3 bottle.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>
    # Apply color splash to video using the last weights you trained
    python3 bottle.py splash --weights=last --video=<URL or path to file>
"""

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
import warnings
warnings.filterwarnings("ignore")

# Read the JSON configuration file
with open('API.json', 'r') as config_file:
    config = json.load(config_file)

# Retrieve the Wandb API key
wandb_api_key = config['wandb_api_key']
wandb.login(key=wandb_api_key)

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
import evaluate as eval

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################
EPOCHS = 1
VAL_STEPS = 1

class CustomConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT ; 1 * 1 = 1
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = VAL_STEPS

    BACKBONE = "resnet101"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 16  # Background + furniture objects

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.5

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    
    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

############################################################
#  Dataset
############################################################

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

        temp_cnt = 0

        for i in images:
            image_id = i['file_name']
            image_path = dataset_dir + '/' +image_id
            height = i['height']
            width = i['width']
            identity = i['id']

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

            temp_cnt+=1
            if(temp_cnt==3):
                break

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


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()


    # Create an EarlyStopping callback
    early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
    
    config = CustomConfig()
    config_dict = config.to_dict()
    config_dict['Epochs'] = EPOCHS

    # Create an Wandb Callback
    wandb_callback = CustomWandbCallback()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=EPOCHS,
                layers='heads',
                custom_callbacks=[early_stopping_callback, wandb_callback])


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


class CustomWandbCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)  # Log the metrics to W&B

if __name__ == '__main__':
    ############################################################
    #  Training
    ############################################################
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    myrun = wandb.init(
                        project='Furniture Segmentation',#project name
                        group='Test',#set group name
                        name='Test1',#set run name
                        resume=False#resume run
                        )

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
        
    ############################################################
    #  Evaluation
    ############################################################
    dataset_path = '../dataset/'
    log_path = './logs/'
    model_path = './logs/object/furniture_segment.h5'

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

    # create config
    config = eval.PredictionConfig()

    # define the model
    model = modellib.MaskRCNN(mode="inference", model_dir=log_path, config=config)

    # load model weights
    model.load_weights(model_path, by_name=True)
        
    # evaluate model on training dataset
    train_mAP, train_precision, train_recall, train_f1_score, train_iou, train_dice = eval.evaluate_model(dataset_train, model, config)
    print("Train mAP, Precision, Recall, F1, IOU, Dice:", train_mAP, train_precision, train_recall, train_f1_score, train_iou, train_dice)
    visualize.plot_actual_vs_predicted("Train", dataset_train, model, config, n_images=3)

    # evaluate model on test dataset
    val_mAP, val_precision, val_recall, val_f1_score, val_iou, val_dice = eval.evaluate_model(dataset_val, model, config)
    print("Val mAP, Precision, Recall, F1, IOU, Dice:", val_mAP, val_precision, val_recall, val_f1_score, val_iou, val_dice)
    visualize.plot_actual_vs_predicted("Val", dataset_val, model, config, n_images=3)

    # evaluate model on test dataset
    test_mAP, test_precision, test_recall, test_f1_score, test_iou, test_dice = eval.evaluate_model(dataset_test, model, config)
    print("Test mAP, Precision, Recall, F1, IOU, Dice:", test_mAP, test_precision, test_recall, test_f1_score, test_iou, test_dice)
    visualize.plot_actual_vs_predicted("Test", dataset_test, model, config, n_images=3)

    wandb.log({"Train mAP": train_mAP, "Val mAP": val_mAP, "Test mAP": test_mAP,
            "Train Precision": train_precision, "Val Precision": val_precision, "Test Precision": test_precision,
            "Train Recall": train_recall, "Val Recall": val_recall, "Test mean Recall": test_recall,
            "Train F1": train_f1_score, "Val F1": val_f1_score, "Test mean F1": test_f1_score,
            "Train IOU": train_iou, "Val IOU": val_iou, "Test IOU": test_iou,
            "Train Dice": train_dice, "Val Dice": val_dice, "Test Dice": test_dice})

    wandb.save(model_path)

    wandb.finish()


# python final.py train --dataset=../dataset/ --weights=coco --logs=./logs