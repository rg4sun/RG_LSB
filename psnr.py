import numpy as np 
from cv2 import cv2 as cv 
import math 

def psnr(img_1, img_2):
    mse = np.mean((img_1 / 1.0 - img_2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

if __name__ == "__main__":
    imgCoverPath = input('cover image path:')
    imgStegoPath = input('stego image path:')
    imgCover = cv.imread(imgCoverPath,cv.IMREAD_UNCHANGED)
    imgStego = cv.imread(imgStegoPath,cv.IMREAD_UNCHANGED)
    print('「载体图」和「嵌入水印图」对比算出的psnr：{}'.format(psnr(imgCover,imgStego)))

