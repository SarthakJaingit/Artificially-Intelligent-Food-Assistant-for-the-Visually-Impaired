from fruitDefectDataset import ffile_path
from PIL import Image
import numpy as np
import pandas as pd
import os

'''
Input:
labels_dict_fn: {"image.jpeg", [1, 0], ...} where the nums are class ids
bounding_box_dict_fn: {"image.jpeg": [[10, 20, 50, 100], [30, 40, 70, 87]]} where the lists are bbox values.
imgs_key_fn: image.jpeg
Returns: It converts the labels dict & bbox dict data to a dictionary in coco format that can be dumped into
a json file. This json file is needed in training the effecientdet scripts as shown here:
https://github.com/rwightman/efficientdet-pytorch
'''

def Coco_Create(labels_dict_fn,bounding_box_dict_fn,imgs_key_fn):
  Coco_dict={}
  images_full_list=[]
  annotations_full_list = []
  bbox_id=0

  for index in range(len(bounding_box_dict_fn)):

        sub_annotation_dict={}

        image_key= imgs_key_fn[index]
        img_path = ffile_path(image_key, full_image_file_paths)
        img = Image.open(img_path).convert("RGB")

        width, height = img.size

        images_sub_dict ={}

        images_sub_dict['file_name'] = image_key
        images_sub_dict['id'] = index
        images_sub_dict['height'] = height
        images_sub_dict['width'] = width
        images_full_list.append(images_sub_dict)
        boxesxywh = bounding_box_dict_fn[image_key]

        labels = labels_dict_fn[image_key]

        for i,box_xywh in enumerate(boxesxywh):
            ann_nest_dict={}

            bbox_id+=1

            x1= box_xywh[0]
            w= box_xywh[2]

            y1= box_xywh[1]
            h= box_xywh[2]

            x2 = w + x1
            y2 = h + y1

            boxes_xyxy = [x1,y1,x2,y2]
            area = (y2 - y1) * (x2 - x1)

            box_xywh = [float(box_xywh[0]),
                        float(box_xywh[1]),
                        float(box_xywh[2]),
                        float(box_xywh[3])]

            ann_nest_dict['bbox'] = box_xywh
            ann_nest_dict['id'] = bbox_id
            ann_nest_dict['image_id'] = index
            ann_nest_dict['area'] = int(area)
            ann_nest_dict['iscrowd'] = 0
            ann_nest_dict['category_id'] = labels[i]+1
            annotations_full_list.append(ann_nest_dict)

  Coco_dict['info'] = {
          "year": "",
          "version": "",
          "description": "",
          "contributor": "",
          "url": "",
          "date_created": ""
      }
  Coco_dict['licenses'] = []
  Coco_dict['annotations'] = annotations_full_list
  Coco_dict['images'] = images_full_list
  categories= [{ "id" : 1,
              "name" : 'Apples',
              "supercategory" : 'Fruit'},
              {"id" : 2,
              "name" : 'Strawberry',
              "supercategory" : 'Fruit'},
               {"id" : 3,
              "name" : 'Peaches',
              "supercategory" : 'Fruit'},
              {"id" : 4,
                "name" : 'Tomato',
                "supercategory" : 'Fruit'},
              {"id" : 3,
                "name" : 'Apple_Bad_Spot',
                "supercategory" : 'Fruit'},
              {"id" : 4,
                "name" : 'Strawberry_Bad_Spot',
                "supercategory" : 'Fruit'},
              {"id" : 7,
                "name" : 'Tomato_Bad_Spot',
                "supercategory" : 'Fruit'},
               {"id" : 8,
              "name" : 'Peaches_Bad_Spot',
              "supercategory" : 'Fruit'}]
  Coco_dict['categories'] = categories
  return Coco_dict
