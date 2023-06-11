import os
import sys
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

VAL_STEPS = 1
class PredictionConfig(Config):

    NAME = 'object'
    # simplify GPU config
    GPU_COUNT = 1

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    # BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT ; 1 * 1 = 1
    IMAGES_PER_GPU = 1

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
	
# calculate the mAP for a model on a given dataset
# def evaluate_model(dataset, model, cfg):
#     APs = []

#     for image_id in dataset.image_ids:
#         # load image, bounding boxes and masks for the image id
#         image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id)

#         # convert pixel values (e.g. center)
#         scaled_image = modellib.mold_image(image, cfg)

#         # make prediction - detect return the below
#         # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
#         # class_ids: [N] int class IDs
#         # scores: [N] float probability scores for the class IDs
#         # masks: [H, W, N] instance binary masks
#         yhat = model.detect([scaled_image])

#         print(yhat)

#         # extract results for first sample
#         r = yhat[0]

#         # calculate statistics, including AP
#         AP, _, _, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        
#         # store
#         APs.append(AP)

#     # calculate the mean AP across all images
#     mAP = np.mean(APs)
#     return mAP

def evaluate_model(dataset, model, cfg):
    APs = []
    precisions = []
    recalls = []
    f1_scores = []

    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id)

        # convert pixel values (e.g. center)
        scaled_image = modellib.mold_image(image, cfg)

        # make prediction - detect returns the following:
        # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        # class_ids: [N] int class IDs
        # scores: [N] float probability scores for the class IDs
        # masks: [H, W, N] instance binary masks
        yhat = model.detect([scaled_image])

        print(yhat)

        # extract results for the first sample
        r = yhat[0]

        # calculate statistics, including AP
        AP, precision, recall, f1_score = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])

        print(AP, precisions, recalls, f1_scores, sep='\n\n')
        
        # store individual values
        APs.append(AP)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    # calculate the mean values
    mAP = np.mean(APs)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1_score = np.mean(f1_scores)

    return mAP, mean_precision, mean_recall, mean_f1_score
