import numpy as np 
import base64, json, random
from cv2 import cv2 as cv

def genEmbedBinStream(imgEmbed):
    '''将嵌入图像抽取成比特流'''
    rowScale = imgEmbed.shape[0]
    columnScale = imgEmbed.shape[1]
    binStreamList = []
    for i in range(rowScale):
        for j in range(columnScale):
            if imgEmbed.item(i,j) != 0:
                imgEmbed.itemset((i,j),1)
            binStreamList.append(imgEmbed.item(i,j))
    return binStreamList

def streamNormalize(binStream): 
    '''
    规范化嵌入流，将嵌入流01比调整至1:1
    '''
    # binstarm长度为偶数最佳，若非偶数调整出的01比将与1:1有所偏差（但应该对lsb分析来看影响不大）
    # zeroPos、onePos分别存所有0、1在binStream中的位置序号
    zeroPos = [ pos for pos in range(len(binStream)) if binStream[pos] == 0]
    # zeeoPos = [pos for pos,value in enumerate(binStream) if value == 0]
    # 这种方式遍历更佳
    onePos = [ pos for pos in range(len(binStream)) if binStream[pos] == 1]
    zeroScale = len(zeroPos)
    oneScale = len(onePos)
    # flag 记录 0多还是1多
    flag = 1 if oneScale > zeroScale else 0
    appendScale = abs(oneScale - zeroScale)//2
    key = [flag,] # key保存的时候首先先存一个flag用于标识规范化之前0多还是1多
    if flag: # 1多，则把多出来的1置0
        key+= [i for i in random.sample(onePos,appendScale)]
        for pos in key[1:]: # key剔除首位的flag
            binStream[pos]=0
    else: # 0多，则把多出来的0置1
        key+= [i for i in random.sample(zeroPos,appendScale)]
        for pos in key[1:]:
            binStream[pos]=1
    with open('./keyfile/keyPos.json','w') as fp:
        json.dump(key,fp)
    return binStream

def binReplace(x:int,b:str,pos:int)->int:
    ''' int数值指定二进制位替换0 or 1，pos从右（二进制低位）从1开始计数 '''
    if b not in ['0','1']:
        print('b must be "0" or "1" !')
        return
    x = bin(x)[2:]
    if len(x) != 8: # 不足八位前面补0
        x= '0'*(8 - len(x)) + x 
    # xbinList = [ x[i] for i in range(len(x))]
    xbinList = list(x)
    # print(xbinList)
    xbinList[len(x) - pos] = b
    # print(xbinList)
    return int(''.join(xbinList),2)

def embeding(imgCover,binStreamList,embedZone,bitPlane=1):
    '''
    具体嵌入操作
    '''
    for coordinate, embedBin in zip(embedZone,binStreamList):
        tmp = imgCover.item(coordinate[0],coordinate[1])
        replace = binReplace(tmp,str(embedBin),bitPlane) 
        # 一定要注意embedBin是int的01但是传入的要是str的'0''1'
        # imgCover.itemset(coordinate,replace) # 指定位平面替换要嵌入的比特流
        # a = imgCover[coordinate[0]][coordinate[1]]
        imgCover[coordinate[0]][coordinate[1]]=replace
    return imgCover

# embedZone生成器（完全随机乱序）
def genRandEmbedZone(imgCoverPath,imgEmbedPath): # 做lsb分析时弃用
    '''生成随机的嵌入位置序列，整张imgcover随机选取位置序号'''
    imgCover = cv.imread(imgCoverPath,cv.IMREAD_GRAYSCALE) # 以灰度图方式读取载体图像
    imgEmbed = cv.imread(imgEmbedPath,cv.IMREAD_GRAYSCALE) #以灰度图方式读取嵌入图像
    binStreamScale = len(genEmbedBinStream(imgEmbed))
    rowScale = imgCover.shape[0]
    columnScale = imgCover.shape[1]
    zone = []
    for i in range(rowScale):
        for j in range(columnScale):
            zone.append(tuple([i,j]))
    return random.sample(zone,binStreamScale)

# embedZone生成器（指定区域随机乱序）
def genNormalZone(imgCoverPath,imgEmbedPath): # lsb分析的时候用这个，
    '''
    从imgcover的第0个像素依次取imgEmbed的比特流长度的位置作为嵌入位置
    但是，对其进行位置打乱嵌入，具体看代码
    '''
    imgCover = cv.imread(imgCoverPath,cv.IMREAD_GRAYSCALE) # 以灰度图方式读取载体图像
    imgEmbed = cv.imread(imgEmbedPath,cv.IMREAD_GRAYSCALE) #以灰度图方式读取嵌入图像
    binStreamScale = len(genEmbedBinStream(imgEmbed))
    rowScale = imgCover.shape[0]
    columnScale = imgCover.shape[1]
    zone = [] # zone是获得整个imgcover的所有像素坐标，可以不这么写，这么写效率比较低，但是懒得优化了
    for i in range(rowScale):
        for j in range(columnScale):
            zone.append(tuple([i,j]))
    return random.sample(zone[:binStreamScale],binStreamScale) # 返回打乱位置的位置序列列表
    # return zone[:binStreamScale] # 直接依次嵌入，不做随机处理

def LSBembedding(imgCoverPath:str,imgEmbedPath:str,embedZone:list,bitPlane=1): 
    '''
    LSB嵌入主函数，imgCoverPath为载体图像路径，imgEmbedPath为水印路径，
    embedZone为嵌入位置（由我写的算法自动生成），bitPlane用于指定嵌入的位平面，默认为1（LSB位）
    '''
    imgCover = cv.imread(imgCoverPath,cv.IMREAD_GRAYSCALE) # 以灰度图方式读取载体图像
    imgEmbed = cv.imread(imgEmbedPath,cv.IMREAD_GRAYSCALE) #以灰度图方式读取嵌入图像
    # 将嵌入水印抽取成比特流并做规范化，将比特流01比例调整为1:1，并做随机话处理，可以理解为对水印加密了
    binStreamList = streamNormalize(genEmbedBinStream(imgEmbed)) 
    # 判断嵌入信息是否过大
    if len(binStreamList) > imgCover.shape[0]*imgCover.shape[1]:
        print('嵌入的信息过大')
        return
    imgStego = embeding(imgCover,binStreamList,embedZone,bitPlane)
    imgCoverName = imgCoverPath[9:].split('.')[0]
    size = int(imgEmbed.shape[0]*imgEmbed.shape[1]/(512*512)*100)
    cv.imwrite('./img/{}_stego{}.bmp'.format(imgCoverName,size),imgStego)
    print('LSB Embeding done!')
    return imgStego

if __name__ == "__main__":
    imgCoverPath = input('Please input coverImage:')
    imgEmbedPath = input('Pleae input embedImage: ')
    bitPlane = int(input('BitPlane='))
    # embedZone = genRandEmbedZone(imgCoverPath,imgEmbedPath) # 做lsb分析时弃用
    embedZone = genNormalZone(imgCoverPath,imgEmbedPath) # 做lsb分析时启用
    with open('./keyfile/zone.json','w') as fp:
        json.dump(embedZone,fp)
    LSBembedding(imgCoverPath,imgEmbedPath,embedZone,bitPlane=bitPlane)

