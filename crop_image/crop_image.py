import lxml
from lxml import etree
import tensorflow as tf
import cv2



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
    cv2.imshow('original', origin_image)
    cv2.imshow('cropped', crop_image)
    cv2.imshow('resized', resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
