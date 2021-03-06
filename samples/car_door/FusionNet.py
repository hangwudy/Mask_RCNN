# coding: utf-8
# # FusionNet for Car Door Detection and Pose Estimation
# @author: Hang Wu
# @date: 2018.12.20


import os
import sys
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import csv
from skimage.color import gray2rgb
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import argparse
import imutils
import pickle
import cv2
import re

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '../../'
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
import time
from mrcnn import visualize_car_door
import mrcnn.model_new as modellib

# PoseNet
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
                default="/home/hangwu/Repositories/Model_output/Attitude_CNN/attitude_vgg16_random.h5",
                # required=True,
                help="path to trained Attitude CNN model model")
ap.add_argument("-mm", "--model_m",
                default="/media/hangwu/TOSHIBA_EXT/Dataset/weights/mask_rcnn_car_door_0250.h5",
                # required=True,
                help="path to trained Mask R-CNN model model")
ap.add_argument("-a", "--latitudebin",
                default="/home/hangwu/Repositories/Model_output/Attitude_CNN/latitude_lb.pickle",
                # required=True,
                help="path to output latitude label binarizer")
ap.add_argument("-o", "--longitudebin",
                default="/home/hangwu/Repositories/Model_output/Attitude_CNN/longitude_lb.pickle",
                # required=True,
                help="path to output longitude label binarizer")
ap.add_argument("-r", "--renderings",
                default="/media/hangwu/TOSHIBA_EXT/Dataset/renderings_square",
                # required=True,
                help="path to input renderings directory")
ap.add_argument("-j", "--valjson",
                default="/media/hangwu/TOSHIBA_EXT/Dataset/annotations/mask_rcnn/car_door_val.json",
                # required=True,
                help="path to validation data annotation directory")
ap.add_argument("-v", "--valdata",
                default="/media/hangwu/TOSHIBA_EXT/Dataset/val_data",
                # required=True,
                help="path to validation data directory")
ap.add_argument("-test", "--test",
                # default="/media/hangwu/TOSHIBA_EXT/Dataset/test_data",
                # default="/home/hangwu/CyMePro/data/test",
                default="/media/hangwu/TOSHIBA_EXT/Dataset/images_from_robot/dataset/images",
                # required=True,
                help="path to test dataset directory")
args = vars(ap.parse_args())

# load the trained convolutional neural network from disk, followed
# by the latitude and longitude label binarizers, respectively
print("[INFO] loading network...")
model2 = load_model(args["model"], custom_objects={"tf": tf})
latitudeLB = pickle.loads(open(args["latitudebin"], "rb").read())
longitudeLB = pickle.loads(open(args["longitudebin"], "rb").read())


def loadim(image_path='Car_Door', ext='png', key_word='car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path, filename)
            image_list.append(image_abs_path)
    return image_list


def pose_estimation(image):

    scale = 500
    output = cv2.resize(image, (scale, scale))
    image_name = image_path.split(os.path.sep)[-1]
    gt_latitude, gt_longitude = gt_name(image_name)
    print("output size:", output.shape)

    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image using Keras' multi-output functionality
    print("[INFO] classifying image...")
    (latitudeProba, longitudeProba) = model2.predict(image)

    # find indexes of both the latitude and longitude outputs with the
    # largest probabilities, then determine the corresponding class
    # labels
    latitudeIdx = latitudeProba[0].argmax()
    longitudeIdx = longitudeProba[0].argmax()
    latitudeLabel = latitudeLB.classes_[latitudeIdx]
    longitudeLabel = longitudeLB.classes_[longitudeIdx]
    """
    # draw the latitude label and longitude label on the image
    latitudeText = "latitude: {} ({:.2f}%)".format(latitudeLabel,
                                                   latitudeProba[0][latitudeIdx] * 100)
    longitudeText = "longitude: {} ({:.2f}%)".format(longitudeLabel,
                                                     longitudeProba[0][longitudeIdx] * 100)
    cv2.putText(output, latitudeText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(output, longitudeText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    print("[INFO] {}".format(latitudeText))
    print("[INFO] {}".format(longitudeText))
    """
    # show the output image
    # cv2.imshow("Output", output)
    image_compare_name = 'car_door_{}_{}.png'.format(latitudeLabel, longitudeLabel)
    image_compare_path = os.path.join(args["renderings"], image_compare_name)
    # print(image_compare_path)
    image_compare = cv2.imread(image_compare_path)
    image_compare = cv2.resize(image_compare, (scale, scale))
    # print("Compare size:", image_compare.shape)

    image_horizontal_concat = np.concatenate((output, image_compare), axis=1)

    plt_fig_save(image_horizontal_concat, 
                image_name[:-4], 
                latitudeLabel, 
                longitudeLabel, 
                gt_latitude, 
                gt_longitude)

    cv2.imwrite("/home/hangwu/Mask_RCNN/detected_images/Results_{:4.0f}.png".format(time.time()), image_horizontal_concat)
    # cv2.imshow("Reality and Prediction", image_horizontal_concat)
    # # cv2.imshow("Comparison", image_compare)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return [image_name[:-4], latitudeLabel, gt_latitude, longitudeLabel, gt_longitude]

def plt_fig_save(img, img_name, pred_latitude, pred_longitude, gt_latitude, gt_longitude):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, ax = plt.subplots(1,1)
    text_on_image = "Prediction: ({},{})\nGround Truth: ({},{})".format(pred_latitude, 
                                                                        pred_longitude, gt_latitude, gt_longitude)
    ax.text(
        0, 1, text_on_image,
        size='small',
        horizontalalignment='left',
        verticalalignment='top',
        # family='serif',
        bbox={'facecolor': 'white', 'alpha':0.5, 'pad':2},
        color='white',
        transform=ax.transAxes
    )
    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.imshow(img)
    plt.savefig("/media/hangwu/TOSHIBA_EXT/Dataset/PoseNet_output/prediction_{}.pdf".format(img_name), 
                dpi=500,
                transparent=True,
                bbox_inches='tight',
                pad_inches=0)


def gt_name(file_name):
    match = re.match(r'([0-9]+)(_+)([0-9]+)(_+)([0-9]+)(\.png)', file_name, re.I)
    latitude = int(match.groups()[0])
    if latitude == 1:
        latitude_fixed = 27
    elif latitude == 2:
        latitude_fixed = 30
    elif latitude == 3:
        latitude_fixed = 18
    elif latitude == 4:
        latitude_fixed = 28
    elif latitude == 5:
        latitude_fixed = 23
    else:
        latitude_fixed = latitude
    longitude = int(match.groups()[4])
    longitude_fixed = 360 - longitude
    if longitude_fixed == 360:
        print("longitude_fixed: ", longitude_fixed)
        longitude_fixed = 0
    
    return latitude_fixed, longitude_fixed
##########################################################################################


# ## Set up logging and pre-trained model paths

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# ## Configuration
# Define configurations for training on the car door dataset.


class CarDoorConfig(Config):
    """
    Configuration for training on the car door dataset.
    Derives from the base Config class and overrides values specific
    to the car door dataset.
    """
    # Give the configuration a recognizable name
    NAME = "car_door"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 (car_door)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 378
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5

    # use resnet101 or resnet50
    BACKBONE = 'resnet101'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50
    PRE_NMS_LIMIT = 6000
    POST_NMS_ROIS_INFERENCE = 500
    POST_NMS_ROIS_TRAINING = 1000


config = CarDoorConfig()
config.display()


# # Define the dataset
class CarPartsDataset(utils.Dataset):

    def load_data(self, annotation_json, images_dir):
        """ Load the coco-like dataset from json
        Args:
            annotation_json: The path to the coco annotations json file
            images_dir: The directory holding the images referred to by the json file
        """
        # Load json from file
        json_file = open(annotation_json)
        car_door_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "car_parts"
        for category in car_door_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
                    class_name))
                return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in car_door_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in car_door_json['images']:
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

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
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


# # Create the Training and Validation Datasets
# In[6]:

dataset_val = CarPartsDataset()
dataset_val.load_data(args["valjson"], args["valdata"])
dataset_val.prepare()


# ## Display a few images from the training dataset

# In[7]:


class InferenceConfig(CarDoorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85


inference_config = InferenceConfig()

# In[8]:


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# In[9]:


# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")

# model_path = model.find_last()
model_path = args["model_m"]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# # Run Inference
# Run model.detect()


# import skimage
real_test_dir = args["test"]

# '/home/hangwu/CyMePro/data/dataset/test'
# '/home/hangwu/CyMePro/data/test'  # '/home/hangwu/CyMePro/data/dataset/test_data'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.JPG']:
        image_paths.append(os.path.join(real_test_dir, filename))


# analysis
M_analysis = []
for image_path in image_paths:
    img = skimage.io.imread(image_path)
    # img = imutils.resize(img, width=360)
    if len(img.shape) < 3:
        img = gray2rgb(img)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]

    image_name = image_path.split(os.path.sep)[-1][:-4]
    if len(r['rois']):
        # xmin ymin
        print('======================================================')
        print('{}: '.format(image_name), r['rois'])
        xmin = r['rois'][:, 1][0]
        ymin = r['rois'][:, 0][0]
        xmax = r['rois'][:, 3][0]
        ymax = r['rois'][:, 2][0]
        xbar = (xmin + xmax) / 2
        ybar = (ymin + ymax) / 2
        center_of_mask = [xbar, ybar]
        print('xmin: {}\nymin: {}\nxmax: {}\nymax: {}'.format(xmin, ymin, xmax, ymax))
        print('Center of the Mask: ', center_of_mask)
        print('======================================================')
    # visualize_car_door.display_instances(img, r['rois'], r['masks'], r['class_ids'],
    #                                      dataset_val.class_names, r['scores'], figsize=(5, 5), image_name=image_name)
    mask_for_pose = visualize_car_door.mask_to_squares(img, r['masks'], xmin, ymin, xmax, ymax)
    # mask_for_pose = visualize_car_door.mask_highlight(img, r['masks'], xmin, ymin, xmax, ymax)

    analysis = pose_estimation(mask_for_pose)
    M_analysis.append(analysis)

# write into a csv
with open("output.csv", 'w') as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow(["Test_Group","Pred_Latitude", "GT_Latitude","Pred_Longitude","GT_Longitude"])
    writer.writerows(M_analysis)
