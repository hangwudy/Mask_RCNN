# Annotation script for 2 car door classes
# Author: Hang Wu
# Date: 2019.07.09

import json
from PIL import Image
import numpy as np
from skimage import measure
# from shapely.geometry import Polygon, MultiPolygon
import os

# EIGEN
from load_image import loadim

# Define which colors match which categories in the images
car_door_first_id = 1
car_door_second_id = 2

is_crowd = 0

# Create the annotations
car_door_annotation = {
    'info': {
        'description': "Car Door Dataset",
        'url': "hangwudy.github.io",
        'version': '0.1',
        'year': 2019,
        'contributor': 'Hang Wu',
        'date_created': '2019/07/09',
    },
    'licenses': [
        {
        "url": "hangwudy.github.io",
        "id": 1,
        "name": 'MIT'
        }
    ],
    "images": [
        {

        }
    ],
    "annotations": [
        {

        }
    ],
    "categories": [
        {
            "supercategory": "car_parts",
            "id": 1,
            "name": 'car_door_1'
        },
        {
            "supercategory": "car_parts",
            "id": 2,
            "name": 'car_door_2'
        }
    ]
}

def create_image_annotation(file_name, height, width, image_id):
    images = {
        'license': 1,
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id
    }
    return images


def create_sub_mask_annotation(is_crowd, image_id, category_id, annotation_id):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)

    annotation = {
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id
    }
    return annotation


def cat_determine(file_name):
    if file_name.find("_cat_1_") != -1:
        car_cat = car_door_first_id
    elif file_name.find("_cat_2_") != -1:
        car_cat = car_door_second_id
    return car_cat


def images_annotations_info(maskpath):

    annotations = []
    images = []

    mask_images_path = loadim(maskpath)
    for id_number, mask_image_path in enumerate(mask_images_path, 1):
        file_name = mask_image_path.split(os.path.sep)[-1][:-4]+'.jpg'

        mask_image = Image.open(mask_image_path)
        category_id = cat_determine(file_name)
        # ID number
        image_id = id_number
        annotation_id = id_number
        # image shape
        width, height = mask_image.size
        # 'images' info 
        image = create_image_annotation(file_name, height, width, image_id)
        images.append(image)
        # 'annotations' info
        annotation = create_sub_mask_annotation(is_crowd, image_id, category_id, annotation_id)
        annotations.append(annotation)
        print('{:.2f}% finished.'.format((id_number / len(mask_images_path) * 100)))
    return images, annotations


if __name__ == '__main__':
    for keyword in ['train', 'val']:
        mask_path = '/home/hangwu/Repositories/Dataset/car_door_mix_annotations/mask_{}'.format(keyword)
        car_door_annotation['images'], car_door_annotation['annotations'] = images_annotations_info(mask_path)
        print(json.dumps(car_door_annotation))
        with open('/home/hangwu/Repositories/Dataset/car_door_mix_annotations/json/car_door_test_{}.json'.format(keyword),'w') as outfile:
            json.dump(car_door_annotation, outfile)
