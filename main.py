#@title Default title text
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:37:23 2018

@author: raffy
"""
import cv2 as cv
import numpy as np
import math


coin_radius = {0:135,1:120,2:107} #keys are as following: 0 is 1 pound, 1 is 1/2 pound and 2 is 1/4 pound

def color_pixel(pixel,coin_class):
    "colors given pixel to red , green or blue according to coin class given"
    if(coin_class==0):
        pixel=[0,0,255] # red
    elif(coin_class==1):
        pixel = [0,255,0] # green 
    elif(coin_class==2):
        pixel=[255,0,0] # blue
    return pixel

def CHT(edges,r):
    "circle hough transform. takes detected edges of image and radius of wanted circles"
    #H = np.zeros((int(edges.shape[0]*resize_ratio) ,int(edges.shape[1]*resize_ratio) ))
    H = np.zeros_like(edges)
    for x in range(edges.shape[0]):
        for y in range(edges.shape[1]):
            if(edges[x,y] == 0):
                continue
            for b in range(y-r,y+r+1):
                if(b<0 or b>=H.shape[1]):
                    continue

                a = (r**2) - ((y-b)**2)
                if(a<0):
                    continue
                
                a1 = int((x - math.sqrt(a)))
                a2 = int((x + math.sqrt(a)))
                
                if(a1<H.shape[0]):
                    H[a1,b]+=1
                if(a2<H.shape[0]):
                    H[a2,b]+=1
     
    
    
    peak = np.amax(H)
    H[H<peak-20] = 0 #image has at most 20 coins so this gives space to give 20 peaks at most
    

    peak_indices = np.argwhere(H>0)
    circles_count = 0
    prev_index = [-5000,-5000]
    for index in peak_indices:
        diff = math.sqrt((index[0] - prev_index[0])**2 + (index[1] - prev_index[1])**2)
        if(diff>=2*r):
            circles_count+=1
        prev_index = index
    return H,circles_count

def mark_coin(original_image,H,coin_class):
    "surrounds 1 pound with red circle, 1/2 pound with green, 1/4 with blue"

    r=coin_radius[coin_class]
    indices = np.argwhere(H>0)
    for x,y in indices :
        for b in range(y-r,y+r+1):
           if(b<0 or b>=H.shape[1]):
               continue
           a = (r**2) - ((y-b)**2)
           if(a<0):
               continue
               
           a1 = int(x - math.sqrt(a))
           a2 = int(x + math.sqrt(a))
                   
           if(a1<original_image.shape[0]):
               original_image[a1,b] = color_pixel(original_image[a1,b],coin_class)
           if(a2<original_image.shape[0]):
               original_image[a2,b] = color_pixel(original_image[a2,b],coin_class)
                   
                   
                    
def classify_coins(path):
       
    
    original_image = cv.imread(path) #read image
    gray_image = cv.cvtColor(original_image, cv.COLOR_RGB2GRAY);  # convert image to grayscale
    median_filtered_image = cv.medianBlur(gray_image, 7); # smoothes image with meian filter of size 7x7
    edges = cv.Canny(median_filtered_image,250,280) #detected edges by canny algorithm
    circles_count = 0
    value = 0

    one_pound_transform , count = CHT(edges,coin_radius[0]) #gets hough transfomr for the one pound
    circles_count+=count # add counted coins to total count
    value += count #add the coins value*their number to the total value of the image

    mark_coin(original_image,one_pound_transform,0) #marks the 1 pounds in the original image
    
    half_pound_transform , count= CHT(edges,coin_radius[1]) #gets hough transfomr for the 1/2 pound
    circles_count+=count
    value += count *0.5
    
    mark_coin(original_image,half_pound_transform,1)
    quarter_pound_transform , count= CHT(edges,coin_radius[2]) #gets hough transfomr for the 1/4 pound
    circles_count+=count
    value += count *0.25
    mark_coin(original_image,quarter_pound_transform,2)
    
    
    return original_image,circles_count,value
    


for i in range(1,9):
    path = 'coins_'+str(i)+'.jpg'
    final_result, circles_count , value= classify_coins(path)
    print("coins_"+str(i)+" image has "+str(circles_count)+" circles")
    print("coins_"+str(i)+" image has value: "+str(value))
    cv.imwrite("./result/classified_coins_"+str(i)+".jpg",final_result ) #write the classified image
