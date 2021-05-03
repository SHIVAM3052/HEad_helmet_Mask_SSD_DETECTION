# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:23:51 2021

@author: Shivam
"""

#%%
#import neccessary libraries
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

# Reading Xml file
tree = ET.parse('C:/Users/Prashant/Downloads/OBJDET/dataset/anno1.xml')
root = tree.getroot()
data = []

# Parsing data for detection
for image in root:
    i = image.tag, image.attrib
    for box in image:
        k = box.tag, box.attrib
        helmet_data = [attribute.text for attribute in box.findall('.//attribute[@name ="has_safety_helmet" ]')]
        mask_data = [attribute.text for attribute in box.findall('.//attribute[@name = "mask"]')]
        if k[1]["label"] == "head":
            id_img = i[1]["id"] + ".jpg"
            width = i[1]["width"]
            height = i[1]["height"]
            xMin = k[1]["xtl"]
            xMax = k[1]["xbr"]
            yMin = k[1]["ytl"]
            yMax = k[1]["ybr"]
            label = k[1]["label"]  
            data.append([id_img,width,height,
                         label,xMax,xMin,yMax,yMin,
                         helmet_data[0],mask_data[0]])

# loading data in dataframe                
df = pd.DataFrame(data)
df = df.rename(columns = {0: 'id_img', 
                            1: 'width',
                            2: 'height',
                            3: 'label',
                            4: 'xMin',
                            5: 'xMax',
                            6: 'yMin',
                            7: 'yMax',
                            8: 'helmet_data',
                            9: 'mask_data'}, inplace = False)

df['mask_data'] = df['mask_data'].replace(['invisible', 'wrong','no'],'no-mask')
df['mask_data'] = df['mask_data'].replace(['yes'],'mask')
df['helmet_data'] = df['helmet_data'].replace(['yes'],'helmet')
df['helmet_data'] = df['helmet_data'].replace(['yes'],'helmet')
df['helmet_data'] = df['helmet_data'].replace(['no'],'no-helmet')
df['label'] = df['label']+"_"+df['helmet_data']+"_"+df['mask_data']
train_df = df.iloc[:1626,:]
train_df = train_df.reset_index()
train_df = train_df.drop(columns=['index'])
test_df = df.iloc[1626:,:]
test_df = test_df.reset_index()
test_df = test_df.drop(columns=['index'])
#%%
import sys
from PIL import Image
#from utils.tfannotation import TFAnnotation
import tensorflow as tf

sys.path.append('C:/Users/Prashant/Downloads/OBJDET/utils/tfannotation.py')

# intiliaze the base path 
BASE_PATH = 'C:/Users/Prashant/Downloads/OBJDET/dataset'

# build path to input training XML files
ANNO_XML = os.pathsep.join([BASE_PATH, "annotations.xml"])

# build the path to the output training and testing record files along with class label file
TRAIN_RECORD = os.pathsep.join([BASE_PATH, "records/training.record"])
TEST_RECORD = os.pathsep.join([BASE_PATH, "records/testing.record"])
CLASSES_FILE = os.pathsep.join([BASE_PATH, "records/classes.pbtxt"])

#intialize the test split size
CLASSES = {"head_helmet_mask":1,"head_helmet_no-mask":2,"head_no-helmet_mask":3,"head_no-helmet_no-mask":4}

path  = 'C:/Users/Prashant/Downloads/OBJDET/dataset/images/'


total = 0   
for i in range(len(train_df)):
    img_path = BASE_PATH +"/images/" + train_df.id_img.iloc[i]
    encoded = tf.io.gfile.GFile(img_path,"rb").read()
    encoded = bytes(encoded)
    
    pilImage = Image.open(img_path)
    (w,h) = pilImage.size[:2]
    
    filename = train_df.id_img.iloc[i]
    encoding = filename[filename.rfind('.')]
    
    label = train_df.label.iloc[i]
    endX = float(train_df.xMax.iloc[i])
    startX = float(train_df.xMin.iloc[i])
    endY = float(train_df.yMax.iloc[i])
    startY = float(train_df.yMin.iloc[i])
    xMin = endX/w
    xMax = startX/w
    yMin = endY/h
    yMax = startY/h
    
    tfAnnot = TFAnnotation()
    tfAnnot.image = encoded
    tfAnnot.encoding = encoding
    tfAnnot.filename = filename
    tfAnnot.width = w
    tfAnnot.height = h
    tfAnnot.xMins.append(xMin)
    tfAnnot.xMaxs.append(xMax)
    tfAnnot.yMins.append(yMin)
    tfAnnot.yMaxs.append(yMax)
    tfAnnot.textLabels.append(label.encode("utf8"))
    tfAnnot.classes.append(CLASSES[label])
    tfAnnot.difficult.append(0)
    
    total+=1
    
    
    
#%%
# Check bounding box on images
import matplotlib.pyplot as plt
import cv2
img = cv2.imread('C:/Users/Prashant/Downloads/OBJDET/dataset/images/0.jpg')
x1 = round(float(train_df.xMax.iloc[1]))
x2 = round(float(train_df.xMin.iloc[1]))
y1 = round(float(train_df.yMax.iloc[1]))
y2 = round(float(train_df.xMin.iloc[1]))
img1 = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),3)
c = plt.imshow(img1)
plt.show()
    
