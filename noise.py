from cv2 import cv2 as cv 
import numpy as np 
import random

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例 
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

if __name__ == "__main__":
    imgPath = input('image path:')
    img = cv.imread(imgPath,cv.IMREAD_UNCHANGED)
    imgName = imgPath[6:].split('.')[0].split('_')[2]
    imgNoised = sp_noise(img,0.001)
    cv.imwrite('./img/{}_noised.bmp'.format(imgName),imgNoised)
    print("Noised image generated!")
    
