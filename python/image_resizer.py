# -*- coding: utf-8 -*-
#
#tool for resizing all images of a folder to specific shape in another folder
#
from resizeimage import resizeimage
from PIL import Image
import os

def resize_images_from_folder(input_folder, output_folder, size) :
    """
    input_folder : folder containing images to resize
    output_folder : folder to store resized images
    last char in input folder and output folder must be / or \\"""
    files = os.listdir(input_folder)
    for file in files :
        try :
            with open(input_folder + file, "r+b") as input :
                with Image.open(input) as full_sized_pic :
                    resized = resizeimage.resize_cover(full_sized_pic, size)
                    resized.save(output_folder + file)
        except : print(file + " failed")
