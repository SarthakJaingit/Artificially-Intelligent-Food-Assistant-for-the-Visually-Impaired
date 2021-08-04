'''
OPTIONAL: This is the class construct for a test file our paper used
to guage how the model did on not predicting anything on random images.

It is not needed to run construct entire pipeline
'''

import glob
from torchvision import transforms
import os
from PIL import Image

class NoiseDataset(object):

  def __init__(self, noise_file_path, size, camera_size):

    self.size = size
    self.noise_file_path = [fp for fp in glob.glob(os.path.join(noise_file_path, "*.JPEG"))]
    self.transforms = transforms.Compose([
                                          transforms.Resize((camera_size, camera_size)),
                                          transforms.ToTensor()])

  def __getitem__(self, idx):

    current_file_path = self.noise_file_path[idx]
    img = Image.open(current_file_path).convert("RGB")

    img = self.transforms(img)
    return img

  def __len__(self):
    if self.size:
      return self.size
    return len(self.noise_file_path)
