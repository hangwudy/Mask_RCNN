import cv2
import numpy as np
import os
import json

# Label file
json_path = "/home/hangwu/Workspace/annotations/car_door_pose.json"
annotation_json = open(json_path)
annotation_list = json.load(annotation_json)


# Dataset file
img_dataset_path = "/home/hangwu/Workspace/Car_Door"

# save path
save_path = "/home/hangwu/Workspace/car_door_square"

_ = 0

for d in annotation_list['annotations']:
    image_name = d.get('image_id')
    image_path = os.path.join(img_dataset_path, image_name)
    latitude = d.get('latitude')
    longitude = d.get('longitude')

    # Bounding Box information >>>
    xmin = d.get('xmin')
    ymin = d.get('ymin')
    xmax = d.get('xmax')
    ymax = d.get('ymax')
    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    side_len_diff_half = int(abs(bbox_height - bbox_width) / 2)

    image = cv2.imread(image_path)

    crop_image = image[ymin:ymax, xmin:xmax]


    if bbox_height >= bbox_width:
        new_patch = np.zeros((bbox_height, bbox_height ,3), np.uint8)
        for row in range(crop_image.shape[0]):
            for col in range(crop_image.shape[1]):
                new_patch[row, col + side_len_diff_half] = crop_image[row, col]
    else:
        new_patch = np.zeros((bbox_width, bbox_width ,3), np.uint8)
        for row in range(crop_image.shape[0]):
            for col in range(crop_image.shape[1]):
                new_patch[row + side_len_diff_half, col] = crop_image[row, col]
    
    cv2.imwrite("{}/{}".format(save_path, image_name), new_patch)
    if _ % 100 == 0:
        print('{:.2f}% finished'.format(_/len(annotation_list['annotations'])*100))
    _ += 1

