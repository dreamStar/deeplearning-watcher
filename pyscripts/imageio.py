# -*- coding: utf-8 -*-
"""
Created on Thu Nov 07 14:28:06 2013

@author: dell
"""

import matplotlib.pyplot as plt
import numpy as np
import os as os
import shelve

class ImageData:
    def __init__(self):
        self.data = np.ndarray((0,0))
        self.size = (0,0)
        self.info = {}
        self.id = -1
        self.name = ""
        self.lindex = -1
        
class Label_converter:
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.label2lindex = {}
        self.lindex2label = {}
        self.catalog_num = 0
        
    def get_lindex(self,label):
        if label in self.label2lindex:
            return self.label2lindex[label]
        else:
            self.label2lindex[label] = self.catalog_num
            self.lindex2label[self.catalog_num] = label
            
            self.catalog_num += 1
           
            return self.catalog_num - 1
            
    def get_label(self,lindex):
        if lindex in self.lindex2label:
            return self.lindex2label[lindex]
        else:
            return ""
            
            
    
    
class ImageIO:
    
    def __init__(self):
        self.imageNumbers = 0
        self.images = []
        self.classDic = {}
        self.basename = ""
        self.label_converter = Label_converter()
        
    def labelParse_exyaleb(self,name):
        ptr = name.find("B")
        return name[ptr+1:ptr+3]
        
    def loadImage(self,path,indexList,imagesize):
        if not os.path.exists(path):
            return False
        data = ImageData()
        
        try:
            data.data = plt.imread(path)
            if imagesize != data.data.shape:
                return False
            data.name = os.path.basename(path)
            data.info = self.infoParse_exyaleb(data.name)
            data.size = data.data.shape
            data.id = self.imageNumbers
            data.lindex = self.label_converter.get_lindex(self.labelParse_exyaleb(data.name))
       
            indexList.append(data.id)
            self.images.append(data)
            self.imageNumbers += 1
            return True
        except IOError:
            print(path)
            return False
        
    def infoParse_exyaleb(self,name):
        ptr = name.find("B")
        person = name[ptr+1:ptr+3]
        ptr = name.find("P")
        pos = name[ptr+1:ptr+3]
        infoDic = {}
        infoDic["person"] = person
        infoDic["pos"] = pos
        ptr = name.find("Ambient")
        if ptr == -1:
            infoDic["ambient"] = True
            return infoDic
        infoDic["ambient"] = False
        
        ptr1 = name.find("A")
        ptr2 = name.find("E")
        ptr3 = name.find(".")
        if ptr3 == -1:
            ptr3 = len(name)
        a = name[ptr1+1:ptr2]
        e = name[ptr2+1:ptr3]
        
        infoDic["a"] = a
        infoDic["e"] = e
        return infoDic
        
    def loadbase_exyaleb(self,path):
        if not (path.endswith("\\") or path.endswith("/")):
            path += "/"
        self.label_converter.clear()
        setNo = ['11','12','13','15','17','20','21','22','23','24','26','27','28']+list(range(29,40))
        #setNo = [11,12,13,15,17,19,20]
        for i in setNo:
            indexList = []
            dicPath1 = "yaleB" + str(i) + "/"
            dicPath1 = path + dicPath1
            dicPath = dicPath1 + "yaleB"+str(i)
            for x in range(0,9):
                infoFile = dicPath + "_P0" + str(x) +".info"
                f = open(infoFile)
                for imagePath in f:
                    imagePath = dicPath1 + imagePath[:-1]
                    self.loadImage(imagePath,indexList,(480,640))
                f.close
                
            self.classDic[self.label_converter.get_lindex(i)] = indexList  
            
    def loadbase_croppedyaleb(self,path):
        if not (path.endswith("\\") or path.endswith("/")):
            path += "/"
        self.label_converter.clear()
        #setNo = [11,12,13,15,17,20,21,22,23,24,26,27,28]+list(range(29,40))
        setNo = ['11','12','13','15','16','17','18','19'\
        ,'20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39']
        for i in setNo:
            indexList = []
            dicPath1 = "yaleB" + i + "/"
            dicPath1 = path + dicPath1
            dicPath = dicPath1 + "yaleB"+str(i)
            
            infoFile = dicPath + "_P00.info"
            f = open(infoFile)
            for imagePath in f:
                imagePath = dicPath1 + imagePath[:-1]
                self.loadImage(imagePath,indexList,(192,168))
            f.close
            self.classDic[self.label_converter.get_lindex(i)] = indexList              
    
    def storeData(self,name):
        db = shelve.open("imageDataBases")
        db[name+"imageNum"] = self.imageNumbers
        db[name+"images"] = self.images
        db.close
    
    def restoreData(self,name):
        db = shelve.open("imageDataBases")
        if not name in db.keys():
            return
        self.imageNumbers = db[name+"imageNum"]
        self.images = db[name+"images"]
                
#if __name__ == "__main__":
    #imageio = ImageIO()
    #imageio.loadbase_exyaleb(r"Z:\share\databases\ExtendedYaleB")
                