#******************************************************************************
#
#  Copyright (c) 2022 Antillia.com TOSHIYUKI ARAI. ALL RIGHTS RESERVED.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#******************************************************************************

import os
import sys
import cv2
import shutil

class YOLOAnnotationWriter(object):

  def __init__(self, yolo_classes_file, images_dir, output_image_path, yolo_output_dir):
    self.images_dir        = images_dir
    self.output_image_path = output_image_path
    self.yolo_output_dir   = yolo_output_dir
    image = cv2.imread(output_image_path)
    self.h, self.w, _ = image.shape
    self.h = float(self.h)
    self.w = float(self.w)
    self.classes =[]
    self.yolo_classes_file = yolo_classes_file

    with open(yolo_classes_file) as f:
      for line in f.readlines():
        self.classes.append(line.strip())

    print("---YOLO classes {}".format(self.classes))


  def getClassIndex(self, cname):
    id = -1
    print("=== getClassIndex cname {} {}".format(cname, self.classes))
    for i, name in enumerate(self.classes):
      if cname == name:
        print("----------{} {} {} ".format(i, cname, name))
        id = i
        break
    return id

  def write(self, detected_objects):
    print("=== YOLOAnnotationWriter write for image {}".format(self.output_image_path))
    if len(detected_objects)==0:
      print("---------No detected objects")
      return 

    SP      = " "
    NL      = "\n"
    basename = os.path.basename(self.output_image_path)
    pos  = basename.find(".jpg")
    name = basename
    if pos >0:
      name = basename[0:pos]

    yolo_annotation_txt = os.path.join(self.yolo_output_dir, name+ ".txt")
    yolo_copied_image   = os.path.join(self.yolo_output_dir, basename)
    org_image_file = os.path.join(self.images_dir, basename)
    shutil.copy2(org_image_file, yolo_copied_image)
    yolo_classes_txt = os.path.join(self.images_dir, "classes.txt")
    shutil.copy2(self.yolo_classes_file, yolo_classes_txt)

    print("=== Writing yolo annotation to: {}".format(yolo_annotation_txt))
    with open(yolo_annotation_txt, mode='w') as f:
      # YOLO annotation format:
      # class_id center_x center_y width height
      # where center_x, center_y, width, height in range[0, 1.0]

      for item in detected_objects:
        #header = "id, class, score, x, y, w, h" 
        print("--- item {}".format(item))
        (_, label,	 confidence, x, y, w, h) = item
        xmin = float(x)
        ymin = float(y)
        xmax = xmin + float(w) 
        ymax = ymin + float(h)
        
        xc = ((xmin + xmax)/2.0)/self.w
        yc = ((ymin + ymax)/2.0)/self.h
        rw  = (xmax - xmin)/self.w
        rh  = (ymax - ymin)/self.h
        id = self.getClassIndex(label)
        line = str(id) + SP + str(xc) + SP + str(yc) + SP + str(rw) + SP + str(rh) + NL
        print(line)
        f.write(line)
       
 