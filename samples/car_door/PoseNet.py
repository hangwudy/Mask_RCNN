# coding: utf-8
# # FusionNet for Car Door Detection and Pose Estimation
# @author: Hang Wu
# @date: 2018.12.20


import os
import sys
import json
import numpy as np
import skimage.io
import time
import re
# from skimage.transform import resize
from skimage.color import gray2rgb
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import tensorflow as tf
import argparse
import imutils
import pickle
import cv2

# Set the MRCNN_DIR variable to the directory of the Mask_RCNN
MRCNN_DIR = '../../'
sys.path.append(MRCNN_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
import time
from mrcnn import visualize_car_door
import mrcnn.model as modellib


# ## Set up logging and pre-trained model paths
# Directory to save logs and trained model
MODEL_DIR = os.path.join(MRCNN_DIR, "output")

# AttitudeNet
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ma", "--model_a",
                default="/home/hangwu/Repositories/Model_output/Attitude_CNN/attitude_cnn.h5",
                # required=True,
                help="path to trained attitude model model")
ap.add_argument("-mm", "--model_m",
                default="/home/hangwu/Repositories/Model_output/Mask_RCNN/mask_rcnn_car_door_0250.h5",
                # required=True,
                help="path to trained Mask R-CNN model model")
ap.add_argument("-la", "--latitudebin",
                default="/home/hangwu/Repositories/Model_output/Attitude_CNN/latitude_lb.pickle",
                # required=True,
                help="path to output latitude label binarizer")
ap.add_argument("-lo", "--longitudebin",
                default="/home/hangwu/Repositories/Model_output/Attitude_CNN/longitude_lb.pickle",
                # required=True,
                help="path to output longitude label binarizer")
ap.add_argument("-r", "--renderings",
                default="/home/hangwu/Repositories/Dataset/renderings_square",
                # required=True,
                help="path to input renderings directory")
ap.add_argument("-vj", "--val_json",
                default="/home/hangwu/Repositories/Dataset/annotations/mask_rcnn/car_door_val.json",
                # required=True,
                help="path to validation data json annotation directory")
ap.add_argument("-vd", "--val_data",
                default="/home/hangwu/Repositories/Dataset/val_data",
                # required=True,
                help="path to validation data directory")
ap.add_argument("-test", "--test",
                # default="/media/hangwu/TOSHIBA_EXT/Dataset/test_data",
                default="/home/hangwu/Repositories/Dataset/real_car_door/dataset/images",
                # required=True,
                help="path to test dataset directory")
ap.add_argument("-codebook", "--codebook",
                # default="/media/hangwu/TOSHIBA_EXT/Dataset/test_data",
                default="/home/hangwu/Repositories/AIBox/annotations/codebook/codebook.json",
                # required=True,
                help="path to code book directory")
args = vars(ap.parse_args())

# load the trained convolutional neural network from disk, followed
# by the latitude and longitude label binarizers, respectively
print("[INFO] loading network...")
model_attitude = load_model(args["model_a"], custom_objects={"tf": tf})
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



# Pose Estimation

# focal length (in terms of pixel) predefinition:
fx_t = 750      # [pixels]
fy_t = 735      # [pixels]
fx_r = 2943.6   # [pixels]
fy_r = 2935.5   # [pixels]
z_r = 1500      # [mm]


def get_diagonal_len(pitch, jaw, code_book):
    """get the diagonal length of the bounding box through Codebook
    """
    with open(code_book) as cb:
        json_data = json.load(cb)
        # Codebook format: {"<latitude_1>": {"<longitude_1>": [<width_1>, <height_1>, <diagonal_length_1>]}}
        diagonal_length = json_data[pitch][jaw]
    return diagonal_length


def f_diagonal_calculation(d, w, h, f_x, f_y):
    """d represents the diagonal length of the bounding box,
    w represents the width, and h represents the height.
    """
    f_diagonal = d / np.sqrt(np.power(w/f_x, 2) + np.power(h/f_y, 2))
    return f_diagonal


def distance_calculation(bbox_t,fx_t,fy_t,pitch,jaw,fx_r,fy_r,z_r):
    """d represents the diagonal length of the bounding box,
    w represents the width, and h represents the height.
    """
    # Unity virtual camera
    # get bbox size
    len_diagonal = get_diagonal_len(pitch, jaw, args["codebook"])
    # width
    w_r = len_diagonal[0]
    # height
    h_r = len_diagonal[1]
    # diagonal
    d_r = len_diagonal[2]
    # f_diagonal calculation
    f_diagonal_r = f_diagonal_calculation(d_r, w_r, h_r, fx_r, fy_r)

    # real camera
    w_t = bbox_t[0]
    h_t = bbox_t[1]
    d_t = bbox_t[2]
    f_diagonal_t = f_diagonal_calculation(d_t, w_t, h_t, fx_t, fy_t)

    # Distance Calculation
    distance_z = z_r * f_diagonal_t / f_diagonal_r * d_r /d_t
    print("==============================================")
    print("estimated distance: {}".format(distance_z))
    print("==============================================")
    return distance_z

def pose_estimation(image, real_longitude, image_name, bbox_info):
    output = imutils.resize(image, width=1000)
    # cv2.imwrite(
    #     "/home/wu/CyMePro/Object_Detection/PoseNet/new_training_set/Results_{}_{}.png".format(image_name, time.time()),
    #     output)
    # pre-process the image for classification
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image using Keras' multi-output functionality
    print("[INFO] classifying image...")
    (latitudeProba, longitudeProba) = model_attitude.predict(image)

    # find indexes of both the latitude and longitude outputs with the
    # largest probabilities, then determine the corresponding class
    # labels
    latitudeIdx = latitudeProba[0].argmax()
    longitudeIdx = longitudeProba[0].argmax()
    latitudeLabel = latitudeLB.classes_[latitudeIdx]
    longitudeLabel = longitudeLB.classes_[longitudeIdx]

    # draw the latitude label and longitude label on the image
    latitudeText = "latitude: {} ({:.2f}%)".format(latitudeLabel,
                                                   latitudeProba[0][latitudeIdx] * 100)
    longitudeText = "longitude: {} ({:.2f}%)".format(longitudeLabel,
                                                     longitudeProba[0][longitudeIdx] * 100)
    # real_longitudeText = "GT longitude: {}".format(real_longitude)
    cv2.putText(output, latitudeText, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(output, longitudeText, (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    # cv2.putText(output, str(real_longitudeText), (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7, (0, 255, 0), 2)

    # display the predictions to the terminal as well
    print("[INFO] {}".format(latitudeText))
    print("[INFO] {}".format(longitudeText))

    # show the output image
    # cv2.imshow("Output", output)
    image_compare_name = 'car_door_{}_{}.png'.format(latitudeLabel, longitudeLabel)
    image_compare_path = os.path.join(args["renderings"], image_compare_name)
    print(image_compare_path)
    image_compare = cv2.imread(image_compare_path)
    image_compare = imutils.resize(image_compare, width=1000)
    print(output.shape[0])
    _ = output.shape[0]

    if not _ == 1000:
        print(output.shape)
        output = cv2.resize(output, (1000, 1000))

    print(image_compare.shape[0])
    _ = image_compare.shape[0]

    if not _ == 1000:
        print(image_compare.shape)
        image_compare = cv2.resize(image_compare, (1000, 1000))

    try:
        image_horizontal = np.hstack((output, image_compare))
        cv2.imwrite("/home/hangwu/Repositories/Dataset/tmp/Results_{}_{:4.2f}.png".format(image_name, time.time()),
                    image_horizontal)
        
        
        # cv2.imshow("Comparison", image_horizontal)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except:
        print("axis not match")
        print(output.shape)
        print(image_compare.shape)
        # cv2.imshow("Detection", output)
        # cv2.imshow("Comparison", image_compare)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    try:
        # Distance Computing
        distance_calculation(bbox_info,fx_t,fy_t,str(latitudeLabel),str(longitudeLabel),fx_r,fy_r,z_r)
    except:
        print("distance calculation failed!")


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

    # All of our training images are 1008x756
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5

    # use resnet101 or resnet50
    BACKBONE = 'resnet101'

    # big object
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 10
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

dataset_val = CarPartsDataset()
dataset_val.load_data(args["val_json"], args["val_data"])
dataset_val.prepare()


# ## Display a few images from the training dataset


class InferenceConfig(CarDoorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.90


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(MRCNN_DIR, ".h5 file name here")

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

image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.JPG']:
        image_paths.append(os.path.join(real_test_dir, filename))

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
        match = re.match(r'([0-9]+)(_+)([0-9]+)(_+)([0-9]+)', image_name, re.I)
        first_element = match.groups()[0]
        if first_element.isdigit():
            real_longitude = 360 - int(match.groups()[4])
        print('test image name: {} ==> Bounding box coordinates '.format(image_name), r['rois'])
        xmin = r['rois'][:, 1][0]
        ymin = r['rois'][:, 0][0]
        xmax = r['rois'][:, 3][0]
        ymax = r['rois'][:, 2][0]
        xbar = (xmin + xmax) / 2
        ybar = (ymin + ymax) / 2
        center_of_mask = [xbar, ybar]

        # Distance Calculation
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        bbox_diagonal = np.sqrt(np.power(bbox_width, 2) + np.power(bbox_height, 2))
        # detected bounding box size
        bbox_t = [bbox_width, bbox_height, bbox_diagonal]

        image_center_x = img.shape[1]/2
        image_center_y = img.shape[0]/2
        center_of_image = [image_center_x, image_center_y]
        print('Center of the image: ', center_of_image)
        # "a" length
        pixel_delta_u = image_center_x - xbar
        # "b" length
        pixel_delta_v = image_center_y - ybar

        print("a: {} pixel; b: {} pixel.".format(pixel_delta_u, pixel_delta_v))

        print('xmin: {}\nymin: {}\nxmax: {}\nymax: {}'.format(xmin, ymin, xmax, ymax))
        print('Center of the Mask: ', center_of_mask)
        print('======================================================')
    visualize_car_door.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                         dataset_val.class_names, r['scores'], figsize=(5, 5), image_name=image_name)
    visualize_car_door.mask_to_squares(img, r['masks'], xmin, ymin, xmax, ymax)
    mask_for_pose = visualize_car_door.mask_to_squares(img, r['masks'], xmin, ymin, xmax, ymax)
    pose_estimation(mask_for_pose, real_longitude, image_name, bbox_t)
