import lxml
from lxml import etree
import tensorflow as tf



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

xml_path = "car_door_0_0.xml"

with tf.gfile.GFile(xml_path, 'r') as fid:
    xml_str = fid.read()
xml = etree.fromstring(xml_str)

xml_dict = recursive_parse_xml_to_dict(xml)

print(xml_dict)



