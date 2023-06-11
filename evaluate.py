import os
import sys
import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "object"
	# number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 16  # Background + furniture objects
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	
# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
    APs = list()

    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, cfg, image_id, use_mini_mask=False)

        # convert pixel values (e.g. center)
        scaled_image = modellib.mold_image(image, cfg)

        # convert image into one sample
        sample = tf.expand_dims(scaled_image, 0)

        # make prediction
        yhat = model.detect(sample, verbose=0)

        # extract results for first sample
        r = yhat[0]

        # calculate statistics, including AP
        AP, _, _, _ = modellib.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        
        # store
        APs.append(AP)

    # calculate the mean AP across all images
    mAP = np.mean(APs)
    return mAP
