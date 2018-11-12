import numpy as np
import yaml
from config import Config
import utils
import random
from PIL import Image, ImageDraw
import skimage.io
import os

'''
def from_yaml_get_class(yaml_path):
    
    with open(yaml_path) as f:
        temp = yaml.load(f.read())
        labels = temp['label_names']
        del labels[0]
    return labels


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "car_door"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1600

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


# config = ShapesConfig()
# config.display()

class CarPartsDataset(utils.Dataset):

    def load_car_door(self):
        # Add class
        self.add_class("car_parts", 1, "car_door")


# color = tuple([random.randint(0, 255) for _ in range(3)])
# print(color)

def get_obj_index(image):
        n = np.max(image)
        return n

# if __name__ == '__main__':
#     yaml_path = 'info.yaml'
#     labels = from_yaml_get_class(yaml_path)
#     print(labels)
#     img = 'car_door_0_12.png'
#     img = Image.open(img)
#     print(img)

#     n = get_obj_index(img)
#     print(n)


instance_masks = []
mask = Image.new('1', (1080, 810))
mask_draw = ImageDraw.ImageDraw(mask, '1')

print(mask_draw)
segmentation = [893.0, 587.5, 897.5, 581.0, 897.5, 537.0, 900.5, 485.0, 899.5, 387.0, 886.5, 343.0, 880.5, 342.0, 881.5, 339.0, 878.5, 334.0, 862.0, 316.5, 813.0, 278.5, 761.0, 243.5, 729.0, 224.5, 703.0, 211.5, 685.0, 204.5, 641.0, 193.5, 594.0, 187.5, 501.0, 183.5, 480.0, 183.5, 477.5, 186.0, 484.5, 208.0, 483.5, 212.0, 485.5, 218.0, 490.5, 224.0, 501.5, 257.0, 500.5, 262.0, 506.5, 272.0, 511.5, 288.0, 515.5, 302.0, 514.5, 307.0, 516.5, 314.0, 519.5, 316.0, 521.5, 324.0, 526.5, 330.0, 536.5, 350.0, 548.5, 382.0, 556.5, 435.0, 558.5, 486.0, 562.5, 519.0, 561.5, 537.0, 564.5, 540.0, 564.5, 552.0, 560.5, 574.0, 560.5, 583.0, 562.0, 584.5, 893.0, 587.5]

mask_draw.polygon(segmentation, fill=1)
print(mask_draw)
bool_array = np.array(mask) > 0

print('bool_array: ',bool_array.shape)
instance_masks.append(bool_array)
print('instance_masks: ', instance_masks)
mask = np.dstack(instance_masks)

print(mask.shape)
# print(mask)


img_path = '/home/hangwu/CyMePro/data/annotations/trimaps_with_window/car_door_0_0.png'
m = skimage.io.imread(img_path).astype(np.bool)
mask = []
mask.append(m)
mask = np.stack(mask, axis=-1)
# print(mask)
print(mask.shape)

'''

PATH_TO_SAVE_IMAGES_DIR = '/home/hangwu/Mask_RCNN/detected_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_SAVE_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# print(os.path.join(PATH_TO_SAVE_IMAGES_DIR, 'image{}.png'.format(i)) for i in range(i))
print(TEST_IMAGE_PATHS)
