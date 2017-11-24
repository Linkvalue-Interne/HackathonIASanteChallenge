import argparse
from skimage import data, io, filters, color
import numpy as np
from skimage.morphology import closing,square
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import gaussian
from skimage.transform import rescale, resize
from os import listdir, path, makedirs
from multiprocessing import Pool
from skimage.transform import rotate

parser = argparse.ArgumentParser(description="Preprocess image by region interest")

parser.add_argument("input_folder", type=str, help="input folder")
parser.add_argument("output_folder", type=str, help="output folder")
parser.add_argument("im_size", type=int, help="final_image_size")
parser.add_argument("pool_size", type=int, help="pool size")


args = parser.parse_args()

def process_folder(fname):
    fname,ext = fname.split(".")
    def raw_seg(im) :
        im_grayscale = color.rgb2gray(im)
        im_red = im_grayscale - np.min(im_grayscale)
        im_norm = np.divide(im_red,np.amax(im_red))
        im_filt = gaussian(im_norm, sigma=10)
        return (im_filt < 0.5).astype(int)

    def expand_square(indexes, max_size, pad):
        return (max(indexes[0] - pad, 0),
                      max(indexes[1] - pad, 0),
                      min(indexes[2] + pad, max_size),
                      min(indexes[3] + pad, max_size))
            

    def find_bbox(label, im) :
        regions = regionprops(label)
        if (len(regions)> 0):
            region = regionprops(label)[np.argmax(regionprops(label))]  
        if(len(regions)> 0 and region.area >= 100 ):
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
        else :
            minr, minc, maxr, maxc = (0,0,label.shape[0],label.shape[1])
            
        minr, minc, maxr, maxc = expand_square((minr, minc, maxr, maxc), label.shape[0],20)
        
        ##mask = np.zeros((label.shape[0],label.shape[1],3))
        ##mask[minr:maxr, minc:maxc] = 1
        return im[minr:maxr, minc:maxc]

        
    def crop_ratio(im):
        h, w, d = im.shape
        newh = int((0.10*h)/2.0)
        neww = int((0.10*w)/2.0)
        return im[newh:-newh, neww:-neww]

    def im_resize(im, size):
        h, w, d = im.shape
        ratio = max(float(size)/float(h), float(size)/float(w))
        im_rescaled = rescale(im, ratio)
        index = np.argmax(im_rescaled.shape)
        crop_size = int(float(im_rescaled.shape[index] - size)/2.0)
        if index == 0 :
            im_crop = im_rescaled[crop_size:-crop_size,:,:]
        else :
            im_crop = im_rescaled[:,crop_size:-crop_size,:]
        
        return resize(im_crop, (size,size,d))

    rot_angles = [0,90,180,270]
    def geometrical_augmentation(im):
        imgs = [im, im[:,::-1,::]]
        imgs_fin = []
        for angle in rot_angles :
            imgs_fin += map(lambda x : rotate(x, angle, mode='reflect'), imgs)

        return imgs_fin

    def normalize(im):
        im_sub_mean = im - im.mean(axis=(0,1))
        im_sub = im_sub_mean - np.min(im_sub_mean)
        im_norm = im_sub/np.amax(im_sub)
        return im_norm
    
    im = im_resize(
        normalize(io.imread(args.input_folder + "/" + fname + "." + ext)), 500
    )
    label = raw_seg(im)
    mask = find_bbox(label, im)
    mask_augmented = geometrical_augmentation(mask)
    for i,m in enumerate(mask_augmented): 
        io.imsave(
            args.output_folder + "/" + fname + "_" + str(i) +"." + ext,
            resize(m,(args.im_size, args.im_size,3))
        )

pool = Pool(args.pool_size)
if not path.exists(args.output_folder):
    makedirs(args.output_folder)

images_name = filter(lambda x: x.endswith(".jpg"), listdir(args.input_folder))
pool.map(process_folder, images_name)
pool.close()
pool.join()


