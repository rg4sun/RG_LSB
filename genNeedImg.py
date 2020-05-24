from cv2 import cv2 as cv
import sys

def genNeedImg(imgPath,size=None,flag='binary'):
    '''
    用于生成指定大小的灰度图或二值图, imgPath为图像路径
    size为tuple类型，用于指定生成图像的尺寸, 如：(512,512)，默认为None表示输出原图像尺寸
    flag为标志转换类型，默认为binary，可选的值为binary或gray
    '''
    imgRow = cv.imread(imgPath)
    if size != None: # 调整图像尺寸
        imgRow= cv.resize(imgRow,size)
    imgGray = cv.cvtColor(imgRow,cv.COLOR_RGB2GRAY) # 转换颜色空间为灰度
    imgName = imgPath[9:].split('.')[0] # 获取图像原始名称
    if flag == 'gray': # 生成灰度图
        cv.imwrite('./images/{}_gray.bmp'.format(imgName),imgGray)
        print('Gray image generated!')
    else: # 生成二值图
        ret, imgBinary = cv.threshold(imgGray,127,255,cv.THRESH_BINARY)
        prop = int(size[0]*size[1]/(512*512)*100) # 以载体图像为512x512，算生成的水印大小占载体图的百分比
        cv.imwrite('./images/{}_binary{}.bmp'.format(imgName,prop),imgBinary) 
        print('Binary image generated!')
        print('threshold:{}'.format(ret)) # 输出转换阈值

if __name__ == "__main__":
    imgName = sys.argv[1]
    size =[int(sys.argv[2]),int(sys.argv[3])]
    flag = sys.argv[4]
    genNeedImg(imgName,tuple(size),flag=flag)
