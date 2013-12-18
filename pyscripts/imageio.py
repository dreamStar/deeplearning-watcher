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

class ImageIO:
    
    def __init__(self):
        self.imageNumbers = 0
        self.images = []
        self.basename = ""
        
        
    def loadImage(self,path):
        data = ImageData()
        data.data = plt.imread(path)
        data.name = os.path.basename(path)
        data.info = self.infoParse_exyaleb(data.name)
        data.size = data.data.shape
        data.id = self.imageNumbers
        self.images.append(data)
        self.imageNumbers += 1
        
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
        #setNo = [11,12,13,15,17,19,20,21,22,23,24,26,27,28]+list(range(29,41))
        setNo = [11]
        for i in setNo:
            dicPath1 = "yaleB" + str(i) + "/"
            dicPath1 = path + dicPath1
            dicPath = dicPath1 + "yaleB"+str(i)
            for i in range(0,9):
                infoFile = dicPath + "_P0" + str(i) +".info"
                f = open(infoFile)
                for imagePath in f:
                    imagePath = dicPath1 + imagePath[:-1]
                    self.loadImage(imagePath)
                f.close
                
    def storeData(self,name):
        db = shelve.open("imageDataBases")
        db[name] = self
        db.close
    
    def restoreData(self,name):
        db = shelve.open("imageDataBases")
        if not name in db.keys():
            return
        self = db[name]
                
if __name__ == "__main__":
    imageio = ImageIO()
    imageio.loadbase_exyaleb(r"D:\databases\ExtendedYaleB")
                