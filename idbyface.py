#-*- coding: utf-8 -*-

import cv2
import sys
from PIL import Image
import pytesseract
import time
import re  

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

print(sys.getdefaultencoding())

#身份证号
r=r'^([1-9]\d{5}[12]\d{3}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])\d{3}[0-9xX])$'

###########根据比例和偏移算出号码位置

def CalcIdRectByFaceRect_normal(x,y,w,h):
    #print(x,y,w,h)
    scale = float(w) / 95
    #print(scale)
    x1 = int(x + (( 0 - 159) ) * scale)
    y1 = int( y + (0 + (149 ) ) * scale)
    x2 = int( x + (0 - 159 + ( 275 ) ) * scale)
    y2 = int(y +(0 + (149 ) + (45 ) ) * scale)
    #print( x1,y1,x2, y2)
    return ( x1,y1,x2, y2)

def CalcIdRectByFaceRect_big(x,y,w,h):
    scale = float(w) / 95
    x1 = int(x + (( 0 - 159) + 10) * scale)
    y1 = int( y + (0 + (149 - 3) ) * scale)
    x2 = int( x + (0 - 159 + ( 275 - 10) ) * scale)
    y2 = int(y +(0 + (149 - 3) + (45 - 10) ) * scale)
    return ( x1,y1,x2, y2)


def CalcIdRectByFaceRect_small(x,y,w,h):
    scale = float(w) / 95
    x1 = int(x + (( 0 - 159) - 10) * scale)
    y1 = int( y + (0 + (149 + 3) ) * scale)
    x2 = int( x + (0 - 159 + ( 275+ 10) ) * scale)
    y2 = int(y +(0 + (149 + 5) + (45 + 10) ) * scale)
    return ( x1,y1,x2, y2)

###########二值化算法
def binarizing(img,threshold):
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if pixdata[x, y] < threshold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


###########去除干扰线算法
def depoint(img):   #input: gray image
    pixdata = img.load()
    w,h = img.size
    for y in range(1,h-1):
        for x in range(1,w-1):
            count = 0
            if pixdata[x,y-1] > 245:
                count = count + 1
            if pixdata[x,y+1] > 245:
                count = count + 1
            if pixdata[x-1,y] > 245:
                count = count + 1
            if pixdata[x+1,y] > 245:
                count = count + 1
            if count > 2:
                pixdata[x,y] = 255
    return img

######## 通过头像的位置 身份证号码识别
def identity_OCR_byFaceRect(oImg,faceRect):
    (x,y,w,h) = faceRect
    iw,ih = oImg.size
    ##将身份证放大3倍
    largeImg = oImg.resize((iw*3,ih*3),Image.ANTIALIAS)
    #largeImg.save('1_large.png')
    
    #print(x,y,w,h)
    (x1,y1,x2,y2) = CalcIdRectByFaceRect_normal(x,y,w,h)
    print("id pos normal: %s,%s,%s,%s"%(x1,y1,x2,y2))
    region = (x1*3,y1*3,x2*3,y2*3) 
    code = GetRegionString(largeImg,region)
    print("code:%s"%code)
    if not re.match(r,code):
        (x1,y1,x2,y2) = CalcIdRectByFaceRect_small(x,y,w,h)
        print("id pos small: %s,%s,%s,%s"%(x1,y1,x2,y2))
        region = (x1*3,y1*3,x2*3,y2*3) 
        code = GetRegionString(largeImg,region)
        print("code:%s"%code)
    if not re.match(r,code):
        (x1,y1,x2,y2) = CalcIdRectByFaceRect_big(x,y,w,h)
        print("id pos big: %s,%s,%s,%s"%(x1,y1,x2,y2))
        region = (x1*3,y1*3,x2*3,y2*3) 
        code = GetRegionString(largeImg,region)
        print("code:%s"%code)
    if not re.match(r,code):
        code = 'no match detect'
    return code, (x1,y1,x2,y2)

def GetRegionString(img,region):
    #裁切身份证号码图片
    cropImg = img.crop(region)
    #cropImg.save('2_crop.png')
    # 转化为灰度图
    grayImg = cropImg.convert('L')
    #grayImg.save('3_grey.png')
    # 把图片变成二值图像。
    bImg =binarizing(grayImg,100)
    #bImg.save('4_bin.png')
    dImg =depoint(bImg)
    #dImg.save('5_depoint.png')
    code = pytesseract.image_to_string(dImg)
    code = PostProc(code)
    return code

######## 号码后处理
def PostProc(s):
    res = s
    res = res.replace(" ","")
    res = res.replace("O","0")
    res = res.replace("U","0")
    res = res.replace("D","0")
    res = res.replace("Z","2")
    res = res.replace("S","5")
    res = res.replace("s","5")
    res = res.replace("o","6")
    res = res.replace("f","7")
    res = res.replace("H","11")
    return res
######## 检测身份证
def DetectFacesAndIDs(window_name, pic_path):
    frame =cv2.imread(pic_path)
    oImg =Image.open(pic_path)

    ih,iw = frame.shape[:2]
    print("image shape:%s,%s"%(ih,iw))
    #人脸识别分类器
    classfier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    
    #识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)    
    color2 = (255, 0, 0)    

    #将当前帧转换成灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                 
    
    #人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    faceRects = classfier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
    if len(faceRects) > 0:            #大于0则检测到人脸                                   
        for faceRect in faceRects:  #单独框出每一张人脸
            x, y, w, h = faceRect      
            print("face: %s,%s,%s,%s"%(x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            code,(x1,y1,x2,y2) = identity_OCR_byFaceRect(oImg,faceRect)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color2, 2)
            #code = code.encode("utf-8")
            #code = PostProc(code)
            #print(u"code:%s"%code)
            print("----------------- detect result: %s"%code)
                    
    #cv2.imshow(window_name, frame)        
    cv2.imwrite("%s.iddet.png"% pic_path,frame)

    
if __name__ == '__main__':
    pic_path =sys.argv[1]
    time1 = time.time()
    DetectFacesAndIDs("detect face area", pic_path)
    time2 = time.time()
    print u'time:' + str(time2 - time1) + 's'
