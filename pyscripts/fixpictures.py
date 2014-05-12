# -*- coding: utf-8 -*-
"""
Created on Wed May 07 09:51:28 2014

@author: dell
"""
import numpy as np
import pickle
import random
class Picture_fixer:
    def __init__(self,size = (28,28)):
        self.h = size[0]
        self.w = size[1]
        
    def fix_picture(self,src):
        xp1 = self._rgb2grey(src)
        xp2 = self._fix_size(xp1)
        
        return xp2
    
    def _rgb2grey(self,src):
        src_h,src_w = src.shape[0:2]
        dst_pic = np.zeros((src_h,src_w))        
        for x in range(src_h):
            for y in range(src_w):
                (r,g,b) = src[x][y]
                dst_pic[x][y] = r*0.3 + g*0.59 + b*0.11
                
        return dst_pic
        
        
    def _fix_size(self,src):
        dst_pic = np.zeros((self.h,self.w))
        src_h,src_w = src.shape[0:2]
        for x in range(self.h):
            for y in range(self.w):
                xp = float(x) / self.h * src_h
                yp = float(y) / self.w * src_w
                dst_pic[x][y] = (src[ceil(xp)][ceil(yp)] + 
                                src[floor(xp)][ceil(yp)] +
                                src[ceil(xp)][floor(yp)] +
                                src[floor(xp)][floor(yp)]) / 4
        return dst_pic
        
        
class Process_datasets:
    def __init__(self,srcdir,size = (28,28),dstdir = r'./datas.pickle'):
        self.initial(size,dstdir)        
                
        
    def initial(self,srcdir,size,dstdir):
        self.dst_dir = dstdir
        self.src_dir = srcdir
        self.fixer = Picture_fixer(size)
        
    def pro(self):
        self.read_all_data()
        self.fix()
        self.split()
        self.random_all()
        self.pik()
        
    def read_all_data(self):
        self.src_data = None
        self.label = None
        
    def split(self):
        self.train_set_x = None
        self.train_set_y = None
        self.valid_set_x = None
        self.valid_set_y = None
        self.test_set_x = None
        self.test_set_y = None
        
    def random_all(self):
        pass
    
    def fix(self):
        self.dataset = np.asarray([map(self.fixer.fix_picture,self.src_data)])
        length = self.dataset.shape[2] * self.dataset.shape[3] 
        tmp = np.zeros((self.dataset.shape[1],length))
        for i in range(self.dataset.shape[1]):
            tmp[i] = self.dataset[0][i].flatten()
        self.dataset = tmp
        
    def pik(self):
        with open(self.dst_dir,'wb') as dataset:
            pickle.dump([[self.train_set_x,self.train_set_y],[self.valid_set_x,self.valid_set_y],[self.test_set_x,self.test_set_y]],dataset)
    
class Process_coil(Process_datasets):
    def __init__(self,srcdir,size = (28,28),dstdir = r'./datas.pickle'):
        self.initial(srcdir,size,dstdir)
        #self.numlist = [4,5,7,10,29,30,34,36,40,58]
        self.numlist = [x+1 for x in range(100)]
        
    def read_all_data(self):
        
        postix = [x for x in range(356) if x % 5 == 0]
        postix = list(map(lambda x:"__"+str(x)+".png",postix))
        self.src_data = []
        self.label = []
        for i in range(len(self.numlist)):
            num = self.numlist[i]
            filenamepre = self.src_dir + "obj" +str(num)     
            for pos in postix:
                filename = filenamepre + pos
                img = imread(filename)
                self.src_data.append(img)
                self.label.append(i)
        
    def split(self):
        type_list = [ [] for x in self.numlist]
        train_num = 45
        valid_num = 15
        test_num = 12
        for i in range(len(self.label)):
            type_list[self.label[i]].append(i)
            
        self.train_set_x = []
        self.train_set_y = []
        self.valid_set_x = []
        self.valid_set_y = []
        self.test_set_x = []
        self.test_set_y = []
        
        for l in type_list:
            random.shuffle(l)
            for i in range(train_num):
                self.train_set_x.append(self.dataset[l[i]])
                self.train_set_y.append(self.label[l[i]])
            for i in range(valid_num):
                i = i + train_num
                self.valid_set_x.append(self.dataset[l[i]])
                self.valid_set_y.append(self.label[l[i]])
            for i in range(test_num):
                i = i + train_num + valid_num
                self.test_set_x.append(self.dataset[l[i]])
                self.test_set_y.append(self.label[l[i]])
                
    def random_all(self):
        def shuffle_set(datas,labels):
            l = range(len(labels))
            random.shuffle(l)
            tmpdata = []
            tmplab = []
            for i in l:
                tmpdata.append(datas[i])
                tmplab.append(labels[i])
            return tmpdata,tmplab
            
        self.train_set_x,self.train_set_y = shuffle_set(self.train_set_x,self.train_set_y)
        self.valid_set_x,self.valid_set_y = shuffle_set(self.valid_set_x,self.valid_set_y)
        self.test_set_x,self.test_set_y = shuffle_set(self.test_set_x,self.test_set_y)