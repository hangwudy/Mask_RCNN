# coding: utf-8

# created by Hang Wu on 2018.10.07
# feedback: h.wu@tum.de

from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import cv2
from numpy import random
import os

# Eigen
import image_overlay
import load_image
import generate_dict

def xml_generator(bndbox, xml_destination_path):
    # Root
    node_root = Element('annotation')
    ## Folder
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = bndbox['folder']
    ## Filename
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = bndbox['filename']
    ## Path
    node_path = SubElement(node_root, 'path')
    node_path.text = bndbox['path']
    ## Source
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'Unknown'
    ## Size
    node_size = SubElement(node_root, 'size')
    ### Width
    node_width = SubElement(node_size, 'width')
    node_width.text = str(bndbox['width'])
    ### Height
    node_height = SubElement(node_size, 'height')
    node_height.text = str(bndbox['height'])
    ### Depth
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(bndbox['depth'])
    ## Segmented
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'
    ## Object
    node_object = SubElement(node_root, 'object')
    ### Name
    node_name = SubElement(node_object, 'name')
    node_name.text = 'car_door'
    ### Pose
    node_pose = SubElement(node_object, 'pose')
    node_pose.text = 'Unspecified'
    ### Truncated
    node_truncated = SubElement(node_object, 'truncated')
    node_truncated.text = '0'
    ### Difficult
    node_difficult = SubElement(node_object, 'difficult')
    node_difficult.text = '0'
    ### Bounding box
    node_bndbox = SubElement(node_object, 'bndbox')
    #### x-y value
    node_xmin = SubElement(node_bndbox, 'xmin')
    node_xmin.text = str(bndbox['xmin'])
    node_ymin = SubElement(node_bndbox, 'ymin')
    node_ymin.text = str(bndbox['ymin'])
    node_xmax = SubElement(node_bndbox, 'xmax')
    node_xmax.text = str(bndbox['xmax'])
    node_ymax = SubElement(node_bndbox, 'ymax')
    node_ymax.text = str(bndbox['ymax'])
    # format display
    xml = tostring(node_root, pretty_print=True)
    xml_name = bndbox['filename'][:-4]+".xml"
    xml_path = os.path.join(xml_destination_path, xml_name)
    fp = open(xml_path, 'w')
    fp.write(xml.decode())
    fp.close()

if __name__ == '__main__':

    # Foreground and background imags
    fg_path = '/home/hangwu/Repositories/Dataset/dataset/car_door_1'
    bg_path = '/home/hangwu/Downloads/val2017'
    
    # Output paths
    xml_dest_path = "/home/hangwu/Repositories/Dataset/dataset/cardoor1/annotation_all/xml"
    image_dest_path = "/home/hangwu/Repositories/Dataset/dataset/cardoor1/car_door_all"
    mask_dest_path = "/home/hangwu/Repositories/Dataset/dataset/cardoor1/annotation_all/mask"
    mask_bw_dest_path = "/home/hangwu/Repositories/Dataset/dataset/cardoor1/annotation_all/mask_bw"

    # Car Door Subcategory: 1 or 2, IMPORTANT for naming the training data
    cd_subcat = 1

    # Test
    test = False
    if test:
        fg_path = '/home/hangwu/Repositories/Mask_RCNN/dataset_tools/Test_Workspace/Image_Generation/Foreground'
        bg_path = '/home/hangwu/Repositories/Mask_RCNN/dataset_tools/Test_Workspace/Image_Generation/Background'
        xml_dest_path = "/home/hangwu/Repositories/Mask_RCNN/dataset_tools/Test_Workspace/Image_Generation/output/xml"
        image_dest_path = "/home/hangwu/Repositories/Mask_RCNN/dataset_tools/Test_Workspace/Image_Generation/output/image"
        mask_dest_path = "/home/hangwu/Repositories/Mask_RCNN/dataset_tools/Test_Workspace/Image_Generation/output/mask"
        mask_bw_dest_path = "/home/hangwu/Repositories/Mask_RCNN/dataset_tools/Test_Workspace/Image_Generation/output/mask_bw"
    
    fg_list = load_image.loadim(fg_path)
    # print(fg_list[1230:1250])
    bg_list = load_image.loadim(bg_path,'jpg','0000')
    # Counter
    progress_show = 1

    for fg_p in fg_list:
        # IMPORTANT: if you want to resize images, don't forget resize in generate_dict
        img_scale = 0.4
        try:
            bnd_info = generate_dict.object_dict(fg_p, img_scale)
            fg = cv2.imread(fg_p, -1)
            # resize the car door images
            fg = cv2.resize(fg, (0,0), fx = img_scale, fy = img_scale, interpolation = cv2.INTER_CUBIC)
            bg_path = random.choice(bg_list)
            bg = cv2.imread(bg_path, -1)
            object_bndbox = image_overlay.overlap(bg, fg, bnd_info, image_dest_path, mask_dest_path, mask_bw_dest_path, cd_subcat)
            xml_generator(object_bndbox, xml_dest_path)
        except:
            print("===========================")
            print(fg_p)
            print(bg_path)
            print("===========================")
        # print(object_bndbox)
        if progress_show % 50 == 0:
            print("++++++++++++++")
            print("{:.2f}%".format(progress_show/len(fg_list)*100))
            print("++++++++++++++")
        progress_show += 1
        