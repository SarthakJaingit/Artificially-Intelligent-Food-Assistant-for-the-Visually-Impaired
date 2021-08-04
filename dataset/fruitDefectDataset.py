'''
This the class construct code that we used to create the dataset
Note: After loading in all the data, we create a dictionary that links every
image id to the the label and bounding box information. We passed this data
to this class.

To utilize this class for further datasets. Just construct a
id_labels: {"image.jpeg", [1, 0], ...} where the nums are class ids
id_bounding_boxes {"image.jpeg": [[10, 20, 50, 100], [30, 40, 70, 87]]} where the lists are bbox values.
'''

import glob
import cv2
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from xml.etree import ElementTree

'''
Input:
image_id: "image.jpeg"
full_image_file_paths: "/content/Folder/images.jpeg"
Returns: It converts the images.jpeg to the full
image filepath that can be loaded.
'''
def ffile_path(image_id, full_image_file_paths):
  for image_path in full_image_file_paths:
    if image_id in image_path:
      return image_path

'''
Input:
image_id: bbox value: [x,y,x,y]
Returns: It returns area of bbox
'''

def find_area_bb(bb_coord):
  bb_coord = bb_coord.numpy()
  area_of_each_bb = list()
  for pair_of_coord in bb_coord:
    area_of_each_bb.append(
        (pair_of_coord[2] - pair_of_coord[0]) * (pair_of_coord[3] - pair_of_coord[1])
    )
  return torch.tensor(area_of_each_bb, dtype=torch.int32)

'''
Input:
image_id: bbox value: [x,y,x,y]
Returns: It converts COCO format [x, y, w, h] bbox to Pascal VOC [x, y, x, y]
'''

def convert_min_max(bb_coord):
  for pair_of_coord in bb_coord:
    pair_of_coord[2], pair_of_coord[3] = (pair_of_coord[0] + pair_of_coord[-2]), (pair_of_coord[1] + pair_of_coord[-1])
  return bb_coord

''' class construct code '''

class FruitDetectDataset(object):
  def __init__(self, id_labels, id_bounding_boxes, transforms, mode, noisy_dataset_path = None, open_image_dir = None, class_list = classes):

    assert len(id_labels) == len(id_bounding_boxes)
    assert sorted(id_labels.keys()) == sorted(id_bounding_boxes.keys())
    self.imgs_key = sorted(id_labels.keys())

    if noisy_dataset_path:
      self.noisy_fp = [fp for fp in glob.glob(os.path.join(noisy_dataset_path, "*.JPEG"))]
      print("Noisy Has been subsetted")
      #Go to this code if you want to subset.
      self.noisy_fp = self.noisy_fp[:40]
    else:
      print("Dataset getting configured without noise loader")
      self.noisy_fp = list()

    self.open_image_info = list()
    self.OI_classes = [name.lower() for name in class_list]

    # np.random.shuffle(self.imgs_key)
    if (mode == "train"):
      self.imgs_key = self.imgs_key[:int(len(self.imgs_key) * 0.8)]
      if open_image_dir:
        for folder_categ in open_image_dir:
          temporary_info = [fp for fp in glob.glob(os.path.join(folder_categ, "pascal" , "*.xml"))]
          # For open images it is actually xml files
          self.imgs_key.extend(temporary_info)
          self.open_image_info.extend(temporary_info)
      if noisy_dataset_path:
        print("Extended {} noisy images to train set".format(int(len(self.noisy_fp) * 0.8)))
        self.imgs_key.extend(self.noisy_fp[:int(len(self.noisy_fp) * 0.8)])
    elif (mode == "test"):
      self.imgs_key = self.imgs_key[int(len(self.imgs_key) * 0.8):]
      if noisy_dataset_path:
        print("Extended {} noisy images to test set".format(int(len(self.noisy_fp) * 0.2)))
        self.imgs_key.extend(self.noisy_fp[int(len(self.noisy_fp) * 0.8):])
    else:
      raise ValueError("Invalid Mode choose from train or test")

    self.id_labels = id_labels
    self.id_bounding_boxes = id_bounding_boxes
    self.full_image_file_paths = glob.glob("/content/Fruit Defects Dataset /Train/*/*/*.jpeg")

    self.transforms = transforms

  def __getitem__(self, idx):

    img_key = self.imgs_key[idx]
    if img_key in self.noisy_fp:
      img_path = img_key
      boxes = torch.zeros((0, 4), dtype=torch.float32)
      labels = torch.as_tensor([], dtype = torch.int64)
    elif img_key in self.open_image_info:
      # img key is an xml file here

      xml_doc = ElementTree.parse(img_key)
      annotation_root = xml_doc.getroot()

      img_path = annotation_root.find("path").text
      bounding_box_nodes = xml_doc.findall("object/bndbox")
      labels_node = xml_doc.findall("object/name")

      boxes, labels = list(), list()

      for node in bounding_box_nodes:
        xmax = node.find("xmax").text
        xmin = node.find("xmin").text
        ymax = node.find("ymax").text
        ymin = node.find("ymin").text

        boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])

      for node in labels_node:
        label = self.OI_classes.index(node.text)
        labels.append(label)

      boxes = torch.as_tensor(boxes, dtype = torch.float32)
      labels = torch.as_tensor(labels, dtype = torch.int64)

    else:
      img_path = ffile_path(self.imgs_key[idx], self.full_image_file_paths)
      boxes = convert_min_max(torch.as_tensor(self.id_bounding_boxes[self.imgs_key[idx]], dtype=torch.float32))
      labels = torch.as_tensor(self.id_labels[self.imgs_key[idx]], dtype=torch.int64)

    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    image_id = torch.tensor([idx])
    area = find_area_bb(boxes)
    iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = image_id
    target["area"] = area
    target["iscrowd"] = iscrowd

    #Query about transforms for labels of images
    if self.transforms:
      sample = {
                'image': img,
                'bboxes': target['boxes'],
                'labels': labels
            }

      sample = self.transforms(**sample)
      img = sample['image']

      if img_key not in self.noisy_fp:
        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)


    return img, target

  def __len__(self):
    return len(self.imgs_key)
