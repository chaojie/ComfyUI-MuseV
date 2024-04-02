# -*- coding: UTF-8 -*-

"""
__author__ = zhiqiangxia
__date__ = 2020-04-15
"""

import xml.etree.ElementTree as ET
import xmltodict


def read_xml2json(path):
    tree = ET.parse(path)
    root = tree.getroot()
    xmlstr = ET.tostring(root).decode()
    dct = xmltodict.parse(xml_input=xmlstr)
    return dct
