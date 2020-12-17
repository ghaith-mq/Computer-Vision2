#!/usr/bin/env python
# coding: utf-8

# In[68]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv
import io
import matplotlib.pyplot as plt
from skimage.exposure import  equalize_hist
from skimage.morphology import dilation, erosion, area_closing, area_opening
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.feature import canny
from skimage.transform import rescale, ProjectiveTransform, warp
from skimage.morphology import dilation, disk
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage
import joblib
from mnist import MNIST
from skimage import img_as_ubyte

# In[36]:


def predict_image(image):
    rf = joblib.load('/autograder/submission/semi.joblib')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.blur(gray,(5,5))
    threshad = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, 
                                          cv.THRESH_BINARY, 199, 5) 
    outer=cv.bitwise_not(threshad,threshad)
    outer_dilated = dilation(outer)
    areaArray = []
    count = 0
    ids=[]
    contours, _ = cv.findContours(outer_dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        area = cv.contourArea(c)
        areaArray.append(area)
        if(area>500000) & (area<10000000):
            ids.append(i)
            count+=1    
            
    def order_points(pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates (tr,tl,br,bl)
        return rect

    def four_point_transform1(image, points):
        rect = order_points(points)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))


        dst = np.array([[0, 0],
        [0, maxWidth - 1],
        [ maxHeight - 1,maxWidth - 1],
        [ maxHeight - 1,0]], dtype = "float32")

        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(image, M, (maxHeight,maxWidth))
        return warped

    ids=np.array(ids)
    flag=1   #this flag is used to point that only one contour detected,
    if(len(ids)==1):  #it is unset if more than one contour were detected so it needs filering
        flag=0
    sud_count=0
    out= np.zeros((gray.shape[0],gray.shape[1]))
    for cnt_id in ids:
        perimeter = cv.arcLength(contours[cnt_id], True)
        epsilon = 0.04* perimeter
        approx = cv.approxPolyDP(contours[cnt_id],epsilon,True)
        app=approx[:,0][:]
        perimeter = cv.arcLength(contours[cnt_id], True)
        points=app
        warped= four_point_transform1(gray, points)
        if(abs(warped.shape[0]-warped.shape[1]) >350) & flag:    #param can be adjusted
            continue
        sud_count+=1
        h=warped.shape[0]
        w=warped.shape[1]
        warped = cv.transpose(warped)
        cv.fillPoly(out, pts=[points], color=(255,255,255))
    ###########################################################
    #Digit Recognizing:
    def center_image(image):
      height, width = image.shape
      wi=(width/2)
      he=(height/2)

      ret,thresh = cv.threshold(image,95,255,0)

      M = cv.moments(thresh)
      if(M["m00"]==0):
            M["m00"]=1
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])

      offsetX = (wi-cX)
      offsetY = (he-cY)
      T = np.float32([[1, 0, offsetX], [0, 1, offsetY]]) 
      centered_image = cv.warpAffine(image, T, (width, height))
      T2 = np.float32([[1.4, 0, -6], [0, 1.6, -10]]) 
      centered_image = cv.warpAffine(centered_image, T2, (width, height))

      return centered_image
    def inverse_img(img):
        return np.int16(255 - img)
    def get_table_only(img, kernel_size=180):
        kernel1 = np.ones((1, kernel_size), dtype='uint8')
        kernel2 = np.ones((kernel_size, 1), dtype='uint8')

        a1 = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel1)
        a2 = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel2)

        a3 = np.min([a1, a2], axis=0)
        return a3
    def four_point_transform(image, points):
        rect = order_points(points)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        f=1
        if(abs(maxHeight-maxWidth) <300):    #param can be adjusted
            f=0
            side=max(maxHeight,maxWidth)
            dst = np.array([
            [0, 0],
            [0, side - 1],
            [ side - 1,side - 1],
            [ side - 1,0]], dtype = "float32")

            M = cv.getPerspectiveTransform(rect, dst)
            warped = cv.warpPerspective(image, M, (side,side))
            return warped,f
        else:
            warped=image
            return warped,f
    ids=np.array(ids)
    sdl=[]
    flag=1   #this flag is used to point that only one contour detected,
    if(len(ids)==1):  #it is unset if more than one contour were detected so it needs filering
        flag=0
    sud_count=0
    for cnt_id in ids:
        perimeter = cv.arcLength(contours[cnt_id], True)
        epsilon = 0.04* perimeter
        approx = cv.approxPolyDP(contours[cnt_id],epsilon,True)
        app=approx[:,0][:]
        perimeter = cv.arcLength(contours[cnt_id], True)
        points=app
        warped,f= four_point_transform(image, points)
        if(f) & flag:    #param can be adjusted
            continue 
        sud_count+=1
        h=warped.shape[0]
        w=warped.shape[1]
        warped = cv.transpose(warped)
        div=h
        ful=w
        gray=cv.cvtColor(warped,cv.COLOR_BGR2GRAY)
        cut=gray[0:div][0:ful]
        ret,cut = cv.threshold(cut,127,255,cv.THRESH_BINARY)
        kernel=np.ones((5,5))
        img_table = get_table_only(np.float32(cut))
        a4 = inverse_img(cut)
        a5 = inverse_img(img_table)
        a6 = inverse_img(np.clip(a4 - a5, 0, None))
        a6=dilation(a6,kernel)
        a6=dilation(a6)
        a6=dilation(a6)
        s=a6.shape[0]//9
        c=np.ones((9,9),dtype='int16')
        c*=-1
        for i in range (9):
            for j in range (9):
                im=a6[i*s :(i+1)*s,j*s:(j+1)*s]
                im=255-im
                if(im.sum()<30):
                    c[i][j]=-1
                else:
                    xs=rescale(im,(28/im.shape[0],28/im.shape[1]), preserve_range=True,anti_aliasing=True)
                    xs=np.uint16(xs)
                    xs = img_as_ubyte(xs)
                    a=center_image(xs)
                    a=erosion(a)
                    a=dilation(a)
                    col=a.ravel()
                    col=col.reshape(-1,1)
                    c[i][j]=rf.predict(col.T)
                    if(c[i][j]==0):
                        c[i][j]=-1
        sdl.append(c)     
#     sudoku_digits = [
#         np.int16([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
#                   [-1, -1, -1,  8,  9,  4, -1, -1, -1],
#                   [-1, -1, -1,  6, -1,  1, -1, -1, -1],
#                   [-1,  6,  5,  1, -1,  9,  7,  8, -1],
#                   [-1,  1, -1, -1, -1, -1, -1,  3, -1],
#                   [-1,  3,  9,  4, -1,  5,  6,  1, -1],
#                   [-1, -1, -1,  8, -1,  2, -1, -1, -1],
#                   [-1, -1, -1,  9,  1,  3, -1, -1, -1],
#                   [-1, -1, -1, -1, -1, -1, -1, -1, -1]]),
#     ]
   # mask = np.bool_(np.ones_like(image))

    # loading train image:
   # train_img_4 = cv2.imread('/autograder/source/train/train_4.jpg', 0)

    # loading model:  (you can use any other pickle-like format)
   # rf = joblib.load('/autograder/submission/random_forest.joblib')

    return out, sdl




# In[74]:


# img=cv.imread("Desktop/Intro to CV/hw2/train_2.jpg")


# In[82]:


# plt.imshow(img)


# In[79]:


# mask, digits = predict_image(img)


# In[81]:


# plt.imshow(mask)


# In[80]:


# digits


# In[ ]:



