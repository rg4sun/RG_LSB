import numpy as np 
from cv2 import cv2 as cv 
import math 

def falseRate(img1,img2):
    img1_float = np.array(img1,dtype=np.float)
    img2_float = np.array(img2,dtype=np.float)
    return np.mean(abs(img1_float-img2_float))/256

if __name__ == "__main__":
    imgEmbedPath = input('embed img path(orignal watermark):')
    imgExtractPath = input('extract img path(extracted watermark):')
    imgEmbed = cv.imread(imgEmbedPath,cv.IMREAD_UNCHANGED)
    imgExtract = cv.imread(imgExtractPath,cv.IMREAD_UNCHANGED)
    print('「提取出来的水印」的错误率：{}'.format(falseRate(imgEmbed,imgExtract)))
    