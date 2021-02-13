# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 09:56:46 2015

@author: JMS
"""

import numpy as np
import cv2
import os
import random
from os import listdir
from os.path import isfile, join

SZ = 256


class charReader:
   tags = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ';
   def __init__(self):
    self.ofile = []
    self.tag = []
    
   def load_digits(self,mypath):
        i = 0;
        for di in listdir(mypath):
            adir = mypath+'\\'+di;           
            onlyfiles = [ f for f in listdir(adir) if isfile(join(adir,f)) ]
            label = self.tags[i];            
            for el in onlyfiles:
                self.ofile.append(adir+'\\'+el);
                self.tag.append(label);
            i= i+1;
        return self.ofile,self.tag;

   def load_digits_by_name(self,mypath):
        i = 0;
        onlyfiles = [ join(mypath,f) for f in listdir(mypath) if isfile(join(mypath,f)) ]
        labels = 0*len(onlyfiles);
        return onlyfiles,labels;



def merge_images(im1,im2):
    mask, contours, hierarchy = cv2.findContours(im1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)            
    letter = im1[y:y+h,x:x+w]
    

    for i in range(h):
        for j in range(w):        
             if(letter[i][j]>0):
                 letter[i][j] = 255 
    
    letter = cv2.resize(letter, (60, 100));

    
    size = 256, 256, 1
    m = np.zeros(size, dtype=np.uint8)
    
    for i in range(100):
        for j in range(60):
              if(letter[i][j]>0):
                    m[95+i][35+j] = 255      
     
    mask, contours, hierarchy = cv2.findContours(im2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)            
    letter = im2[y:y+h,x:x+w]
    
    for i in range(h):
        for j in range(w):        
             if(letter[i][j]>0):
                 letter[i][j] = 255 
    letter = cv2.resize(letter, (100, 150));  
          

    for i in range(150):
        for j in range(100):
              if(letter[i][j]>0):
                    m[50+i][100+j] = 255      
    

    return m


def file_to_img(fn):
    print('loading "%s" ...' % fn)
    digits = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    ret,thresh = cv2.threshold(digits,127,255,0)
    digits = cv2.resize(thresh, (SZ, SZ)) # normalize to 128 by 128
    return digits

    
gxlabels="0123456789ABCDEFGHIJK"
gylabels="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

x = np.arange(len(gxlabels))
y = np.arange(len(gylabels))

myfolder = os.getcwd();

myfolder2 = myfolder+"\\original_letters\\"
minusfolder = myfolder2+"minus\\"
plusfolder = myfolder2+"plus\\"

myfolder = myfolder+"\\Synthetic1\\"

for i in x:
   for j in y:
       label = gxlabels[i]+gylabels[j]


chread = charReader();
digits_minus, labels = chread.load_digits_by_name(minusfolder)      
digits_minus = map(file_to_img,digits_minus)
i = -1;
for el1 in digits_minus:
      i = i+1;
      chread2 = charReader();
      digits_plus, labels = chread2.load_digits_by_name(plusfolder)      
      digits_plus = map(file_to_img,digits_plus)  
      j = 0;
      for el2 in digits_plus:
          
          pic=merge_images(el1,el2)
          label = gxlabels[i]+gylabels[j]
          j = j+1;
          myim_name = label+".jpg"
          mydir = myfolder+"\\"+myim_name
          cv2.imwrite(mydir, pic)             
          print("Saving "+mydir)
