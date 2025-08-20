import csv
import cv2
import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

csv_file="./oilannotation/data.csv"
xml_folder="./oilannotation"

xml_files = glob.glob("{}/xmls/*".format(xml_folder))
# print(xml_files)
import re
alist=[]
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

alist=xml_files
alist.sort(key=natural_keys)
# print(alist)
print(len(alist))

heights=[]
widths=[]
xmins=[]
ymins=[]
xmaxs=[]
ymaxs=[]

for i, xml_file in enumerate(alist):
        tree = ET.parse(xml_file)
        #print(xml_file)
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        xmin = int(tree.findtext("./object/bndbox/xmin"))
        ymin = int(tree.findtext("./object/bndbox/ymin"))
        xmax = int(tree.findtext("./object/bndbox/xmax"))
        ymax = int(tree.findtext("./object/bndbox/ymax"))
        #heights.append(height)
        widths.append(width)
        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        heights.append(height)

widths=pd.Series(widths)
xmins=pd.Series(xmins)
xmaxs=pd.Series(xmaxs)
ymins=pd.Series(ymins)
ymaxs=pd.Series(ymaxs)
heights=pd.Series(heights)
df=pd.read_csv("./oilannotation/data.csv",names=["path","height","width","xmin","ymin","xmax","ymax","classtype","class"])
print(df.head())
n = df.columns[1]
print(n)
df.drop(n, axis = 1, inplace = True)
# df[n]="newclass"
# print(df)
df.to_csv('./oilannotation/train.csv',index=False)
newtype=df["classtype"]
newclass=df["class"]
df1=pd.read_csv('./oilannotation/train.csv')
print(df1)
