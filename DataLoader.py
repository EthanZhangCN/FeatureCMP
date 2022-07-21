import os
import cv2
from torch import rand
from utils.RegionProposal import RegionProposal
import random
import torch
import pdb

class DataLoader():
    def __init__(self,path) -> None:
        self.rpn = RegionProposal(min_size=64,debug_mod=False)
        if torch.cuda.is_available(): 
           self.rpn = self.rpn.cuda()
        self.paths = self.getFilePaths(path)
        print("file numbers:",len(self.paths))

    def getFilePaths(self,file_path):
        img_paths = []
        for root,dirs,files in os.walk(file_path):
            for i in range(len(files)):
                if files[i].split('.')[-1]=='jpg':
                    img_paths.append(os.path.join(root,files[i]))
        return img_paths

    def __getitem__(self,index):
        r = random.randint(0,1)
        if r:
            return self.getOneMatch(self.paths[index])
        else:
            return self.getOneDismatch(self.paths[index])

    def getOneMatch(self, path):
        img = cv2.imread(path)
        img = self.resize_512(img)
        img_cp = img.copy()
        rois = self.rpn.predict(img)
        sub_imgs = []
        for i in range(len(rois[:, 0])):
            temp_img =img_cp[int(rois[i, 1]): int(rois[i, 3]), int(rois[i, 0]):int(rois[i, 2]),:]
            sub_imgs.append(temp_img)
            # cv2.imwrite('./test/rois/'+str(i)+'.jpg',temp_img)   

        return img_cp,sub_imgs,1
    
    def getOneDismatch(self,path):
        dis_path = self.getRandomPath(path)
        _,sub_imgs,_ =self.getOneMatch(dis_path)

        img = cv2.imread(path)
        img = self.resize_512(img)
        return img,sub_imgs,0

    def getRandomPath(self,path):
        new_path = random.choice(self.paths)
        if new_path==path and len(self.paths)>1:
           return self.getRandomPath(path)
        return new_path

    def resize_512(self,image):
        nw = 512 if image.shape[1] < image.shape[0] else int(512 * image.shape[1] / float(image.shape[0]))
        nh = 512 if image.shape[0] < image.shape[1] else int(512 * image.shape[0] / float(image.shape[1]))
        image = cv2.resize(image,(nw,nh))
        return image

if __name__=='__main__':
    folder ='/home/zhizizhang/test/B00AMYTXM0/'
    ss = DataLoader(folder)

    # ss.getOnePatch(ss.paths[8])
    # print(ss.paths[1])
    # aaaa = ss.getRandomPath(ss.paths[1])
    # print(aaaa)

    # for i, (a,b,c) in enumerate(ss):
    #     print(i,len(a),len(b),c)

    # for i in range(100):
    #     r = random.randint(0,1)
    #     print(r)

