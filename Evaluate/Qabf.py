# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
import math
import numpy as np
import cv2
class qabf():
    def __init__(self,img1,img2,imgf):
        #model parameters 模型参数
        self.img1=img1.astype(np.float32)
        self.img2 = img2.astype(np.float32)
        self.imgf = imgf.astype(np.float32)
        self.L = 1;
        self.Tg = 0.9994;
        self.kg = -15;
        self.Dg = 0.5;
        self.Ta = 0.9879;
        self.ka = -22;
        self.Da = 0.8;

        #Sobel Operator Sobel算子
        self.h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        self.h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        self.h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        self.result=self.cal_qabf()
    # 数组旋转180度
    def flip180(self,arr):
        new_arr = arr.reshape(arr.size)
        new_arr = new_arr[::-1]
        new_arr = new_arr.reshape(arr.shape)
        return new_arr

    #相当于matlab的Conv2
    def convolution(self,k, data):
        k = self.flip180(k)
        data = np.pad(data, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
        n,m = data.shape
        img_new = []
        for i in range(n-2):
            line = []
            for j in range(m-2):
                a = data[i:i+3,j:j+3]
                line.append(np.sum(np.multiply(k, a)))
            img_new.append(line)
        return np.array(img_new)


    #用h3对strA做卷积并保留原形状得到SAx，再用h1对strA做卷积并保留原形状得到SAy
    #matlab会对图像进行补0，然后卷积核选择180度
    #gA = sqrt(SAx.^2 + SAy.^2);
    #定义一个和SAx大小一致的矩阵并填充0定义为aA，并计算aA的值
    def getArray(self,img):
        SAx = self.convolution(self.h3,img)
        SAy = self.convolution(self.h1,img)
        gA = np.sqrt(np.multiply(SAx,SAx)+np.multiply(SAy,SAy))
        n, m = img.shape
        aA = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                if(SAx[i,j]==0):
                    aA[i,j] = math.pi/2
                else:
                    aA[i, j] = math.atan(SAy[i,j]/SAx[i,j])
        return gA,aA

    #the relative strength and orientation value of GAF,GBF and AAF,ABF;
    def getQabf(self,aA,gA,aF,gF):
        n, m = aA.shape
        GAF = np.zeros((n,m))
        AAF = np.zeros((n,m))
        QgAF = np.zeros((n,m))
        QaAF = np.zeros((n,m))
        QAF = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                if(gA[i,j]>gF[i,j]):
                    GAF[i,j] = gF[i,j]/gA[i,j]
                elif(gA[i,j]==gF[i,j]):
                    GAF[i, j] = gF[i, j]
                else:
                    GAF[i, j] = gA[i,j]/gF[i, j]
                AAF[i,j] = 1-np.abs(aA[i,j]-aF[i,j])/(math.pi/2)

                QgAF[i,j] = self.Tg/(1+math.exp(self.kg*(GAF[i,j]-self.Dg)))
                QaAF[i,j] = self.Ta/(1+math.exp(self.ka*(AAF[i,j]-self.Da)))

                QAF[i,j] = QgAF[i,j]*QaAF[i,j]

        return QAF
    def cal_qabf(self):
    #对strB和strF进行相同的操作
        gA,aA = self.getArray(self.img1)
        gB,aB = self.getArray(self.img1)
        gF,aF = self.getArray(self.imgf)

        QAF = self.getQabf(aA,gA,aF,gF)
        QBF = self.getQabf(aB,gB,aF,gF)

        #计算QABF
        deno = np.sum(gA+gB)
        nume = np.sum(np.multiply(QAF,gA)+np.multiply(QBF,gB))
        output = nume/deno
        return output

#
# img1=cv2.imread(r'C:\Users\dell\Desktop\color\c_01_asr.tif',0)
# img2=cv2.imread(r'C:\Users\dell\Desktop\color\c_01_brw.tif',0)
# imgf=cv2.imread(r'C:\Users\dell\Desktop\color\c_01_cnn.tif',0)
# my_qabf=qabf(img1,img2,imgf)
# print(my_qabf.result)

