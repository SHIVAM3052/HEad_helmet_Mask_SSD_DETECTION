# -*- coding: utf-8 -*-
"""
Created on Sat May  1 16:18:16 2021

@author: Prashant
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
df.to_csv('file1.csv')
#%%

from PIL import Image
from utils.tfannotation import TFAnnotation
import tensorflow as tf



# intiliaze the base path 
BASE_PATH = 'C:/Users/Prashant/Downloads/OBJDET'

# build path to input training XML files
ANNO_XML = os.pathsep.join([BASE_PATH, "annotations.xml"])

# build the path to the output training and testing record files along with class label file
TRAIN_RECORD = BASE_PATH + "/records/training.record"
TEST_RECORD =  BASE_PATH + "/records/testing.record"
CLASSES_FILE =BASE_PATH + "/records/classes.pbtxt"

#intialize the test split size
CLASSES = {"head_helmet_mask":1,"head_helmet_no-mask":2,"head_no-helmet_mask":3,"head_no-helmet_no-mask":4}


total_train = 0
# Training Dataset for tfrecords.
writer_train = tf.io.TFRecordWriter(TRAIN_RECORD)
   
for i in range(len(train_df)):
    
    img_path = BASE_PATH +"/dataset/images/" + train_df.id_img.iloc[i]
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
    
    total_train += 1
    
    features = tf.train.Features(feature=tfAnnot.build())
    example = tf.train.Example(features=features)
    writer_train.write(example.SerializeToString())
writer_train.close()
print("Train record created")

total_test = 0
# Testing Dataset for tfrecords. 
writer_test = tf.io.TFRecordWriter(TEST_RECORD)
   
for i in range(len(test_df)):
    img_path = BASE_PATH +"/dataset/images/" + test_df.id_img.iloc[i]
    encoded = tf.io.gfile.GFile(img_path,"rb").read()
    encoded = bytes(encoded)
    
    pilImage = Image.open(img_path)
    (w,h) = pilImage.size[:2]
    
    filename = test_df.id_img.iloc[i]
    encoding = filename[filename.rfind('.')]
    
    label = test_df.label.iloc[i]
    endX = float(test_df.xMax.iloc[i])
    startX = float(test_df.xMin.iloc[i])
    endY = float(test_df.yMax.iloc[i])
    startY = float(test_df.yMin.iloc[i])
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
    
    total_test += 1
    
    features = tf.train.Features(feature=tfAnnot.build())
    example = tf.train.Example(features=features)
    writer_test.write(example.SerializeToString())
writer_test.close()
print("Test record created")

#%%

    
    
    
    
    