import lxml
from lxml import etree
import tensorflow as tf
import cv2
import numpy as np


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    # if not xml:

    if not len(xml):
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def bndbox_from_xml(xml_path):
    with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    xml_dict = recursive_parse_xml_to_dict(xml)
    # bndbox_xy = []
    bbox = {}
    xmin = xml_dict['annotation']['object'][0]['bndbox']['xmin']
    ymin = xml_dict['annotation']['object'][0]['bndbox']['ymin']
    xmax = xml_dict['annotation']['object'][0]['bndbox']['xmax']
    ymax = xml_dict['annotation']['object'][0]['bndbox']['ymax']
    # bndbox_xy.append(int(xmin))
    # bndbox_xy.append(int(ymin))
    # bndbox_xy.append(int(xmax))
    # bndbox_xy.append(int(ymax))
    file_path = xml_dict['annotation']['path']
    bbox['path'] = file_path
    bbox['xmin'] = int(xmin)
    bbox['ymin'] = int(ymin)
    bbox['xmax'] = int(xmax)
    bbox['ymax'] = int(ymax)
    # print(bndbox_xy)
    return bbox


if __name__ == "__main__":
    bndbox= bndbox_from_xml('car_door_0_0.xml')
    origin_image = cv2.imread(bndbox['path'])

    crop_image = origin_image[bndbox['ymin']:bndbox['ymax'], bndbox['xmin']:bndbox['xmax']]
    resize_image = cv2.resize(crop_image, (227, 227))
    # bbox width and height
    img_width = bndbox['xmax'] - bndbox['xmin']
    img_heigth = bndbox['ymax'] - bndbox['ymin']
    # left or upon part
    edge_to_minus = round(abs(img_heigth - img_width) / 2)
    if img_width > img_heigth:
        long_edge = img_width
        edge_to_plus = img_width - img_heigth - edge_to_minus
        bndbox['ymin'] -= edge_to_minus
        bndbox['ymax'] += edge_to_plus
    else:
        
        long_edge = img_heigth
        print(long_edge)
        edge_to_plus = img_heigth - img_width - edge_to_minus
        print(img_width)
        print(edge_to_minus)
        print(edge_to_plus)
        bndbox['xmin'] -= edge_to_minus
        bndbox['xmax'] += edge_to_plus
        print(bndbox['xmax']-bndbox['xmin'])
        print(bndbox['xmax'])

    crop_image2 = origin_image[bndbox['ymin']:bndbox['ymax'], bndbox['xmin']:bndbox['xmax']]
    resize_image2 = cv2.resize(crop_image2, (227, 227))
    print(crop_image2.shape[0])
    print(crop_image2.shape[1])
    # here is not equal because out of the range of the original image

    # for extrem situation can the above method exeed the range
    blanck_image = np.zeros((long_edge, long_edge, 3), np.uint8)
    cv2.imshow("test2", blanck_image)
    
    for row in range(crop_image.shape[0]):
        for col in range(crop_image.shape[1]):
            if img_width > img_heigth:
                blanck_image[row + edge_to_minus, col] = crop_image[row, col]
            else:
                blanck_image[row, col + edge_to_minus] = crop_image[row, col]
    
    cv2.imshow("test3", blanck_image)





    cv2.imshow('original', origin_image)
    cv2.imshow('cropped', crop_image)
    cv2.imshow('resized', resize_image)
    cv2.imshow('cropped2', crop_image2)
    cv2.imshow('resized2', resize_image2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
