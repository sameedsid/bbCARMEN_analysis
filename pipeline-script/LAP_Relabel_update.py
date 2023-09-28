# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 2020
@author: David Stirling updated by Rebecca Senft
Script for colouring a set of segmentation label images according to an object tracking number.
This script takes in a table of objects and a uint16 segmented image.
Resulting images are saved in a new folder.
"""

import skimage
import skimage.io
import matplotlib.colors
from skimage.color import label2rgb
import numpy as np
import pandas as pd
import math
import os
from pathlib import Path
import imageio


# Give me paths to the objects table, images table and input/output folders.
# The folders must already exist.
output = "output-path"
cellsfile = os.path.join(output, "MyExpt_AcceptedBeads.csv")
imagesfile = os.path.join(output, "MyExpt_Image.csv")
segpath = os.path.join(output, "label_map","") #use the last "" to get a trailing slash added
outputpath = os.path.join(output, "relabel_output","")
labelfieldname = 'TrackObjects_Label'
wantRGB = True
wantGif = True # only if wantRGB==True. Do you want to save a gif of the RGB images?
groupBy = 'Metadata_Lane' # column to group images by (e.g., a column to indicate fov for which you have multiple timepoints)

if not os.path.exists(outputpath):
    os.mkdir(outputpath)
    
print("Loading data")

main_df = pd.read_csv(cellsfile)
image_df = pd.read_csv(imagesfile)
meta_df = image_df[['ImageNumber', 'FileName_BeadsOnFluor', groupBy]]
main_df = main_df.merge(meta_df, on="ImageNumber")

#default color list: 
#color_list = np.array(['red', 'blue', 'yellow', 'magenta', 'green','indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen'] * 200)
#I want to use modified glasbey:
color_list = np.array(['#9a6901', '#953f1f', '#855c89', '#aed1d4', '#ae083f', '#a0e491', '#f2cdfe','#ff3464','#01753f','#d1afee',
                        '#bc9157','#ac567c','#00fdcf','#00d493','#03ff75','#bceddb','#80793d','#da005d','#da005d','#b501fe',
                        '#a795c5','#53823b','#9cfeff','#9ee1ff','#c84248','#db7e00','#feaaf6','#d342fc','#b3723b','#ff6201', '#f9ff00', 'red', 'blue', "green", "cyan", "yellow"])
my_colormap = [matplotlib.colors.to_rgb(color) for color in color_list]*200
gif_images = []

print("Generating Image List")
imagelist = image_df['FileName_Bead_seg'].tolist()
generatedname = [segpath + i for i in imagelist]
image_df['segfile'] = generatedname
print("Processing images")
for index, row in image_df.iterrows():
    inputimg = skimage.io.imread(row['segfile'])
    outputimg = np.zeros_like(inputimg, dtype='uint16')
    if int(row['Count_AcceptedBeads']) > 0:
        objectstolabel = main_df[main_df['ImageNumber'] == row['ImageNumber']]
        for index2, eachobject in objectstolabel.iterrows():
            boolarray = inputimg == int(eachobject['Number_Object_Number'])
            newnumber = eachobject[labelfieldname]
            if not math.isnan(newnumber):
                outputimg[boolarray] = eachobject[labelfieldname]
    save_path = os.path.join(outputpath,row[groupBy])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    skimage.io.imsave(os.path.join(save_path, row['FileName_BeadsOnFluor']), outputimg, check_contrast=False)
    print("Saved " + row['FileName_Bead_seg'])
    if wantRGB:
        rgb_save_path = os.path.join(outputpath,row[groupBy]+"_rgb")
        if not os.path.exists(rgb_save_path):
            os.mkdir(rgb_save_path)
        uniquelabels = np.unique(outputimg)[1:]
        #colorstouse = my_colormap[uniquelabels]
        colorstouse = [my_colormap[coloridx] for coloridx in uniquelabels]
        labelled = label2rgb(outputimg, image=None, colors=colorstouse, bg_label=0, bg_color=(0, 0, 0), kind='overlay')
        scale_label = skimage.img_as_ubyte(labelled)
        saveName = row['FileName_BeadsOnFluor'].split(sep=".")[0]
        skimage.io.imsave(os.path.join(rgb_save_path, saveName + '_rgb.png'), scale_label, check_contrast=False)
    
#save out gif
if wantRGB and wantGif: 
    folderlist = [folder[0] for folder in os.walk(outputpath) if "rgb" in folder[0]]
    for folder in folderlist: 
        images = []
        filenames = os.listdir(folder)
        filename_without_extension = Path(folder).stem
        for filename in filenames:
            images.append(imageio.imread(os.path.join(folder, filename)))
        imageio.mimsave(os.path.join(outputpath, filename_without_extension+".gif"), images, duration=0.3)

print("Script complete!")
