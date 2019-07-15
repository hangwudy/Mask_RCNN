# -*- coding: UTF-8 -*-
# 2019/02/13 by HANG WU

import shutil
import os
import re


def loadim(image_path = 'images', ext = 'png', key_word = 'car_door'):
    image_list = []
    for filename in os.listdir(image_path):
        if filename.endswith(ext) and filename.find(key_word) != -1:
            current_path = os.path.abspath(image_path)
            image_abs_path = os.path.join(current_path,filename)
            image_list.append(image_abs_path)
    return image_list

def copy_and_rename(original_path, destination_path):
    file_name = os.path.split(original_path)[-1]
    match = re.match(r'([A-Za-z_]+)(_+)([0-9]+)(_+)([0-9]+)(\.png)', file_name, re.I)
    latitude = int(match.groups()[2])
    longitude = int(match.groups()[4])
    longitude_fixed = longitude - 250
    if longitude_fixed < 0:
        longitude_fixed += 360
    new_name = "car_door_{}_{}.png".format(latitude, longitude_fixed)
    new_file_path = os.path.join(destination_path, new_name)
    shutil.copy2(original_path, new_file_path)
    # print("Original path:", original_path)
    # print("New path:", new_file_path)
    if longitude_fixed < 0:
        print("Something goes wrong!!!!!")



def copy_and_rename_anticlockwise(original_path, destination_path):
    file_name = os.path.split(original_path)[-1]
    match = re.match(r'([A-Za-z_]+)(_+)([0-9]+)(_+)([0-9]+)(\.png)', file_name, re.I)
    latitude = int(match.groups()[2])
    longitude = int(match.groups()[4])
    longitude_fixed = longitude - 270
    if longitude_fixed < 0:
        longitude_fixed += 360
    longitude_final = 360 - longitude_fixed
    latitude_final = 90 - latitude
    new_name = "car_door_{}_{}.png".format(latitude_final, longitude_final)
    new_file_path = os.path.join(destination_path, new_name)
    shutil.copy2(original_path, new_file_path)
    # print("Original path:", original_path)
    # print("New path:", new_file_path)
    if longitude_final < 0:
        print("Something goes wrong!!!!!")


if __name__ == "__main__":
    dataset_path = "/home/hangwu/Repositories/Dataset/Tuer_2_all"
    dest_path = "/home/hangwu/Repositories/Dataset/dataset/car_door_2"
    image_list = loadim(image_path=dataset_path)
    i = 0
    for image_p in image_list:
        copy_and_rename_anticlockwise(image_p, dest_path)
        i += 1
        percentage = round(i/len(image_list)*100, 2)
        if i%1000 == 0:
            print("{}% finished".format(percentage))
            