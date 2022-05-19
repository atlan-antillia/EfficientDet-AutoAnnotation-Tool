# Copyright 2022 antillia.com Toshiyuki Arai 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# PascalVOC2YOLOConverter.py

import glob

import os
import pickle
import xml.etree.ElementTree as ET
from os import listdir, getcwd
from os.path import join
import shutil
import traceback


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def xml_to_yolo(input_file, output_file):
    classes = ["Pedestrian_Signal_Blue",
      "Pedestrian_Signal_Red", 
      "Traffic_Signal_Blue", 
      "Traffic_Signal_Red",
      "Traffic_Signal_Yellow"]

    tree = ET.parse(input_file)
    root = tree.getroot()
    size = root.find('size')
    w    = int(size.find('width').text)
    h    = int(size.find('height').text)

    with open(output_file, "w") as file:
      for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')



def convert_to_yolo(input_dir, output_dir):
    print("---- input_dir {}".format(input_dir))
    image_files = glob.glob(input_dir + "/*.jpg")
    for image_file in image_files:
      basename = os.path.basename(image_file)
      name     = basename
      pos      = basename.find(".jpg")
      if pos>0:
        name = basename[0:pos]

      xml_anno_file  = os.path.join(input_dir, name + ".xml")
      yolo_anno_file = os.path.join(output_dir,"signals_" + name + ".txt")
      out_imagefile  = os.path.join(output_dir, "signals_" +name + ".jpg")
      shutil.copy2(image_file, out_imagefile)

      xml_to_yolo(xml_anno_file, yolo_anno_file)

  

if __name__ == "__main__":
  try:
    input_dir  = "./valid_xml"
    output_dir = "./valid"
    if not os.path.exists(input_dir):
      raise Exception("Not found " + input_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    convert_to_yolo(input_dir, output_dir)

  except:
    traceback.print_exc()
