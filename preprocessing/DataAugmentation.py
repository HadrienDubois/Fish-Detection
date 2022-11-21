from pyclbr import Class
from tkinter import image_names
import cv2 as cv
import numpy as np
import random as rd
import os

SEED = 42
rd.seed(SEED)


    
def RandomTranslate(img):
    x = rd.randint(-150,150)
    y = rd.randint(-150,150)
    TransMatrix = np.float32([[1,0,x],[0,1,y]])
    dim = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, TransMatrix, dim)


def RandomRotation(img, rotPoint = None) :
    (height, width) = img.shape[:2]
    angle = rd.randint(-360,360)
    
    if rotPoint is None :
        rotPoint = (height//2, width//2)
    
    RotMatrix = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dim = (width, height)    
    return cv.warpAffine(img, RotMatrix, dim)


def RandomFlip(img):
    x = rd.randint(-1,1)
    return cv.flip(img,x)

def RandomColorConversion(img) :
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray
    

def HorizontalFlip(img):
    return cv.flip(img,1)

def ResizeImage(img):
        width = 320
        height = 320
        dimensions = (width, height)
        return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
    
def ResizeAllImages(path):
    for img in os.listdir(path) :
        cv_image = cv.imread(path+img)
        cv.imwrite(path+'REsized '+img, ResizeImage(cv_image))
        os.remove(path+img)



def Data_Augmentation(PATH):
    for img in os.listdir(PATH):
        cv_img = cv.imread(PATH+img)
        #cv.imwrite(PATH+'translated '+img, RandomTranslate(cv_img))
        #cv.imwrite(PATH+'rotated '+img,RandomRotation(cv_img))
        cv.imwrite(PATH+'flipped '+img, HorizontalFlip(cv_img))
        #cv.imwrite(PATH+'gray '+img, RandomColorConversion(cv_img))


PATH1 ='photos/amphiprioninae/'

PATH_list = [PATH1] #PATH2]

for path in PATH_list :
    ResizeAllImages(path)
    #Data_Augmentation(path)
    
