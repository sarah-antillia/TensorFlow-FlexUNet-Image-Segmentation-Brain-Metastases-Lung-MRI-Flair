# Copyright 2026 antillia.com Toshiyuki Arai
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
# ImageMaskDatasetGenerator.py
# 2026/11/06 

import os
import csv
import glob
import cv2
import nibabel as nib

import shutil
from PIL import Image,  ImageOps
import numpy as np
import traceback

class ImageMaskDatasetGenerator:

  def __init__(self, size=512, augmentation=True):
    self.RESIZE = (size, size)

    self.CORE_RGB_COLOR  = (200,20,20)
    self.WHOLE_RGB_COLOR = (255,255,0)

    self.file_format = ".png"
    self.index = 10000 
    self.augmentation = augmentation
    if self.augmentation:
      self.hflip    = True
      self.vflip    = True
 
  def crop(self, image):
    h, w = image.shape[:2]
    if w > h:
        y1 = 0
        y2 = h
        x1 = (w - h)//2
        x2 = x1 + h
        image = image[y1:y2,x1:x2]
        return image
    elif w <= h:
        x1 = 0
        x2 = w
        y1 = (h - w )//2
        y2 = y1 + w
        image = image[y1:y2,x1:x2]
        return image

  def normalize(self, image):
     min1, max1 = image.min(), image.max()
     if max1 > min1:
        image = (image - min1) / (max1 - min1) * 255.0
     else:
        image = image * 0
     return image


  def colorize_mask(self, mask, rgb_color):
    h, w = mask.shape[:2]
    colorized = np.zeros((h, w, 3), dtype=np.uint8) 
    (r, g, b) = rgb_color
    colorized[np.equal(mask, 1)] = (b, g, r)
    return colorized
   
  def generate(self, dataset_dir, output_images_dir, output_masks_dir): 
     subdirs = os.listdir(dataset_dir)
     for subdir in subdirs:
        self.index += 1       

        fullsubdir = os.path.join(dataset_dir, subdir)
        # Flair image files
        image_files     = glob.glob(fullsubdir + "/*_flair_img.nii")
        core_seg_files  = glob.glob(fullsubdir + "/*_core_seg.nii")
        whole_seg_files = glob.glob(fullsubdir + "/*_whole_seg.nii")
        l1 = len(image_files)
        l2 = len(core_seg_files)
        l3 = len(whole_seg_files)
        if l1 == 0 or l2 == 0 or l3 == 0:
          print("Skipped: Unmatched the number of files")
          continue
        image_file     = image_files[0]
        core_seg_file  = core_seg_files[0]
        whole_seg_file = whole_seg_files[0]

        image_data     = nib.load(image_file)
        images = image_data.get_fdata()
        print("images_shape ", images.shape)

        core_seg_data = nib.load(core_seg_file)
        core_segs = core_seg_data.get_fdata()
        
        print("core_segs ", core_segs.shape)

        whole_seg_data = nib.load(whole_seg_file)
        whole_segs = whole_seg_data.get_fdata()
        print("whole_seg ", whole_segs.shape)
        
        h,  w,  num  = images.shape
        hc, wc, numc = core_segs.shape
        hw, ww, numw = whole_segs.shape
        if h != hc or h != hw:
          print("Skipped: Unmached height case")
          continue
        if w != wc or w != ww:
          print("Skipped: Unmached width case")
          continue

        for n in range(num):
          image = images[:,:,n]
          core  = core_segs[:,:,n]
          whole = whole_segs[:,:,n] 
          if image.any() == 0:
            print("Skipped an empty image------------")
            continue
          if whole.any() > 0 or core.any() > 0:
            filename = str(self.index) + "_" + str(n) +  self.file_format
            out_imagefilepath = os.path.join(output_images_dir, filename )   
            image = self.normalize(image)
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = self.crop(image)
            image = cv2.resize(image, self.RESIZE)
            cv2.imwrite(out_imagefilepath, image)
            print("Savd {}".format(out_imagefilepath))
 
            colorized_core  = self.colorize_mask(core,  self.CORE_RGB_COLOR)
            colorized_whole = self.colorize_mask(whole, self.WHOLE_RGB_COLOR)

            colorized = colorized_core + colorized_whole
            colorized = cv2.rotate(colorized, cv2.ROTATE_90_COUNTERCLOCKWISE)

            colorized = self.crop(colorized)
            colorized = cv2.resize(colorized, self.RESIZE)

            out_maskfilepath = os.path.join(output_masks_dir, filename )   
            cv2.imwrite(out_maskfilepath, colorized)
            print("Savd {}".format(out_maskfilepath))
            if self.augmentation:
              self.augment(image,     filename, output_images_dir)
              self.augment(colorized, filename, output_masks_dir  )

          else:
            print("Skipped an empty mask------------")
         

  def augment(self, image, basename, output_dir):
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

     
  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]
     
  
if __name__ == "__main__":
  try:
    dataset_dir  = "./Brain-Mets-Lung-MRI-Path-Segs/"
  
    output_dir   = "./Brain-Metastases-Lung-MRI-master/"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
      os.makedirs(output_dir)
    
    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")
  
    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)
  
    generator = ImageMaskDatasetGenerator()
       
    generator.generate(dataset_dir, output_images_dir, output_masks_dir)
    
  except:
    traceback.print_exc()