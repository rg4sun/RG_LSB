import numpy as np 
import base64, json, random
from cv2 import cv2 as cv

def retRowStream(binStream,keyPos):
    '''
    将规范化的嵌入流还原成原始嵌入流（原始水印的01流）
    '''
    flag = keyPos[0]
    if flag:
        for pos in keyPos[1:]:
            binStream[pos]=1
    else:
        for pos in keyPos[1:]:
            binStream[pos]=0
    return binStream

def extractEmbedBinStream(imgStego,embedZone,bitPlane=1):
    '''
    从stego图片中提取嵌入的水印流信息，此处需要embedZone密钥（记录了嵌入位置）
    '''
    embedInt = [ imgStego.item(coordinate[0],coordinate[1]) for coordinate in embedZone] # 提取嵌入像素int值
    binStream = []
    for i in embedInt:
        iStr = bin(i)[2:]
        if len(iStr) != 8:
            iStr = '0'*(8-len(iStr)) + iStr
        b = iStr[len(iStr) - bitPlane]
        binStream.append(int(b))
    return binStream
    

def LSBextracting(imgStegoPath,embedZone,imgEmbedSize,keyPos,bitPlane=1):
    '''
    LSB提取主函数，embedZone密钥（记录了嵌入位置，imgEmbedSize为原始水印尺寸，
    keyPos是用于恢复原始水印比特流的密钥
    '''
    imgStego = cv.imread(imgStegoPath,cv.IMREAD_UNCHANGED)
    # 从stego提取嵌入的binstream（这里提取出来的stream是规范化的，01比为1:1）
    binStreamList = extractEmbedBinStream(imgStego,embedZone,bitPlane=1)
    # 将规范化的stream逆回成原始binstream，此时得到的才是真正嵌入的信息流
    binStreamList = retRowStream(binStreamList,keyPos)
    embedList = [] # 将01流恢复成二值图像素列表
    for b in binStreamList:
        if b == 1:
            embedList.append(255)
        else:
            embedList.append(0)
    # 利用像素列表恢复成图像矩阵
    imgEmbedMatrix = [ embedList[i*imgEmbedSize[0]:(i+1)*imgEmbedSize[0]] for i in range(imgEmbedSize[1])]
    # print(imgEmbedMatrix)
    imgEmbed = np.array(imgEmbedMatrix,dtype=np.uint8)
    cv.imwrite('./img/img_extract.bmp',imgEmbed)
    print('LSB extracting done!')


if __name__ == "__main__":
    extractFile = input('Please input extractFile: ')
    zoneFile = input('Please input zonefile: ')
    with open(zoneFile,'r') as fp:
        ezone = json.load(fp)
    keyFile = input('Please input keyfile: ')
    with open(keyFile,'r') as fp:
        keyPos = json.load(fp)
    size = input('Input size like 512x512 : ')
    size = size.split('x')
    size[0] = int(size[0])
    size[1] = int(size[1])
    LSBextracting(extractFile,ezone,size,keyPos,bitPlane=1)

