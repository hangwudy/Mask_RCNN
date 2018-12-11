# coding: utf-8
# @author: Hang Wu
# @date: 2018.12.11
# Car Door Mask Detection

# Combination of mask rcnn and pose net


# In[1]:
import os
import sys
import json
import numpy as np
import time
import skimage.io
import re
from PIL import Image, ImageDraw
# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '/home/hangwu/Mask_RCNN'
sys.path.append(ROOT_DIR) 
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model_new as modellib


# ## Set up logging and pre-trained model paths
# This will default to sub-directories in your mask_rcnn_dir, but if you want them somewhere else, updated it here.
# 
# It will also download the pre-trained coco model.

# In[3]:


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# ## Configuration
# Define configurations for training on the car door dataset.

# In[4]:


class CarDoorConfig(Config):
    """
    Configuration for training on the car door dataset.
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
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256) # (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 


config = CarDoorConfig()
config.display()


# # Define the dataset

# In[5]:


class CarPartsDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """
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
                print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
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
                
    def load_mask(self, image_id):
        """ 
        Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances].
        Args:
            image_id: The id of the image to load masks for
        Returns:
            masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
#         print(image_id) ##
        
        image_info = self.image_info[image_id]
        
#         print(image_info.items()) ##
#         print(image_info['path']) ##
        mask_name = image_info['path'].split(os.path.sep)[-1][:-4]+'.png' ##
#         print(mask_name)
        mask_path = os.path.join('/home/hangwu/CyMePro/data/annotations/trimaps_with_window', mask_name)
#         print(mask_path)
        
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        mask_with_window = []  ##
        mask_ww = []  ##
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)
                
                ##
                instance_masks_ww = skimage.io.imread(mask_path).astype(np.bool)  # #
                mask_ww.append(instance_masks_ww)
                ##

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        ##
        mask_with_window = np.stack(mask_ww, axis = -1)
        
        
#         print(class_ids)
#         print('mask origin: ', mask.shape)
#         print('mask with window: ', mask_with_window.shape)
        return mask_with_window, class_ids  # mask, class_ids
    
    def load_pose(self, image_id):
        """ 
        Load instance poses for the given image.
        Pose format: Latitude, Longitude
        Args:
            image_id: The id of the image to extract pose infomation
        Returns:
            poses: An array of shape [latitude, longitude] with
                one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        
        image_name = image_info['path'].split(os.path.sep)[-1]
        print("Test Pose, image name: {}".format(image_name))
        match = re.match(r'([A-Za-z_]+)(_+)([0-9]+)(_+)([0-9])', image_name, re.I)
        class_name = match.groups()[0]
        latitude = match.groups()[2]
        longitude = match.groups()[4]
        print("latitude: {}".format(latitude))
        print("longitude: {}".format(longitude))



# # Create the Training and Validation Datasets
# In[6]:


dataset_train = CarPartsDataset()
dataset_train.load_data('/home/hangwu/CyMePro/botVision/JSON_generator/output/car_door_train.json',
                        '/home/hangwu/CyMePro/data/dataset/train_data')
dataset_train.prepare()

dataset_val = CarPartsDataset()
dataset_val.load_data('/home/hangwu/CyMePro/botVision/JSON_generator/output/car_door_val.json',
                      '/home/hangwu/CyMePro/data/dataset/val_data')
dataset_val.prepare()


# ## Display a few images from the training dataset

# In[7]:


dataset = dataset_train
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    ##### Test pose >>>>
    dataset.load_pose(image_id)
    ##### Test pose <<<<
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


# # Create the Training Model and Train
# This code is largely borrowed from the train_shapes.ipynb notebook.

# In[17]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


# In[18]:


# Which weights to start with?
# "imagenet", "coco", or "last"
init_with = "coco"  

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# ## Training
# 
# Train in two stages:
# 
# 1. Only the heads. Here we're freezing all the backbone layers
# and training only the randomly initialized layers (i.e. the ones
# that we didn't use pre-trained weights from MS COCO). To train
# only the head layers, pass layers='heads' to the train() function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary,
# but we're including it to show the process. Simply pass layers="all
# to train all layers.
# 
# 

# In[19]:


# Train the head branches

start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=10, 
            layers='heads')
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')


# In[20]:


# Fine tune all layers

start_train = time.time()
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=12, 
            layers="all")
end_train = time.time()
minutes = round((end_train - start_train) / 60, 2)
print(f'Training took {minutes} minutes')


# # Prepare to run Inference
# Create a new InferenceConfig, then use it to create a new model.

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
model_path = model.find_last()

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# # Run Inference
# Run model.detect() on real images.
# 
# We get some false positives, and some misses.
# More training images are likely needed to improve the results.

# In[10]:


# import skimage
real_test_dir = '/home/hangwu/CyMePro/data/test'  # '/home/hangwu/CyMePro/data/dataset/test_data'
image_paths = []
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg', '.JPG']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                dataset_val.class_names, r['scores'], figsize=(5, 5))

