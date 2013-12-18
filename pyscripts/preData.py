# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:34:45 2013

@author: mitc
"""

import numpy as np
import imageio as imio
import random


class PreImageData:
    
    def __init__(self,imageio = 0):
        if imageio == 0:
            self.imageio = imio.ImageIO();
        else:
            self.imageio = imageio

    
    def pre_train_set(self,imageio,index,label_name):
        print len(index),len(imageio.images)
        train_set_x = np.ndarray((len(index),imageio.images[index[0]].data.shape[0]*imageio.images[index[0]].data.shape[1]),dtype = float)
        train_set_y = np.ndarray(len(index),int)
        shape = imageio.images[index[0]].data.shape
        for i in range(len(index)):    
            try:
                train_set_x[i] = imageio.images[index[i]].data.reshape((1,shape[0]*shape[1])) / 255.0
                train_set_y[i] = imageio.images[index[i]].lindex
            except ValueError:
                print imageio.images[index[i]].lindex
                print imageio.images[index[i]].data.shape
                print shape[0],shape[1]
        self.train_set = (train_set_x,train_set_y)
        
    
    def pre_valid_set(self,imageio,index,label_name):
        valid_set_x = np.ndarray((len(index),imageio.images[index[0]].data.shape[0]*imageio.images[index[0]].data.shape[1]),dtype = float)
        valid_set_y = np.ndarray(len(index),int)
        shape = imageio.images[index[0]].data.shape
        for i in range(len(index)):           
            valid_set_x[i] = imageio.images[index[i]].data.reshape((1,shape[0]*shape[1])) / 255.0
            valid_set_y[i] = imageio.images[index[i]].lindex
        self.valid_set = (valid_set_x,valid_set_y)   
        
    def pre_test_set(self,imageio,index,label_name):
        test_set_x = np.ndarray((len(index),imageio.images[index[0]].data.shape[0]*imageio.images[index[0]].data.shape[1]),dtype = float)
        test_set_y = np.ndarray(len(index),int)
        shape = imageio.images[index[0]].data.shape
        for i in range(len(index)):           
            test_set_x[i] = imageio.images[index[i]].data.reshape((1,shape[0]*shape[1])) / 255.0
            test_set_y[i] = imageio.images[index[i]].lindex 
        self.test_set = (test_set_x,test_set_y)
        
        
        
class PreImageData_ExYaleB(PreImageData):
            
    def get_data_set(self,classes,train_set_num,valid_set_num,test_set_num):
        train_index = []
        valid_index = []
        test_index = []
        for class_one in classes:
            lst = self.imageio.classDic[class_one]
            if(len(lst) < train_set_num+valid_set_num+test_set_num):
                continue
            temp = list(range(len(lst)))[0:train_set_num+valid_set_num+test_set_num]
            random.shuffle(temp)
            for i in range(len(temp)):
                if i < train_set_num:
                    train_index.append(lst[temp[i]])
                else :
                    if i < (train_set_num + valid_set_num):
                        valid_index.append(lst[temp[i]])
                    else:
                        test_index.append(lst[temp[i]])
        random.shuffle(train_index)
        random.shuffle(valid_index)
        random.shuffle(test_index)
        self.pre_train_set(self.imageio,train_index,"person")
        self.pre_valid_set(self.imageio,valid_index,"person")
        self.pre_test_set(self.imageio,test_index,"person")
        
    def load_data(self,classes,train_set_num,valid_set_num,test_set_num,dataset = 0):
        if dataset != 0:
            #self.imageio.loadbase_exyaleb(dataset)
            self.imageio.loadbase_croppedyaleb(dataset)
        self.get_data_set(classes,train_set_num,valid_set_num,test_set_num)
        return (self.train_set,self.valid_set,self.test_set)
        
if __name__ == "__main__":
     
    #pre = PreImageData_ExYaleB()
    #pre.load_data([11,12,13],200,100,10,r"Z:\share\databases\ExtendedYaleB")
     pass