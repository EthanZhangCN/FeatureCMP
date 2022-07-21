
from unittest import result
from FeatureCMPModel import FeatureCMPModel
import torch
import torch.nn as nn
import pdb
from DataLoader import DataLoader
import random
import os

def train_step(model,src,tgt,label):
    model.train()
    model.optimizer.zero_grad()
 
    result = model.forward(src,tgt)
    # label = torch.zeros([len(result),1])+label
    # label = label.cuda()
    loss =  model.loss_func.forward(result,label)
    loss.backward()
    model.optimizer.step()
    return loss

class loss_fun(nn.Module):
    def __init__(self) -> None:
        super(loss_fun,self).__init__()
        self.crossentropy = nn.CrossEntropyLoss()
    def forward(self,y_c,y):
        L = - ( y* torch.log(y_c) + (1 - y)*torch.log(1 - y_c) )
        return L

def train(
    train_folder = '/home/zhizizhang/test/B00A87MIE6',
    save_path = './runs/',
    EPOCHS = 20000,
    MAX_BATCH = 8,
    MIN_BATCH = 1,
    SAVE_CYCLE = 2000,
    resume = None,
    ):
    dataloader = DataLoader(train_folder)

    model = FeatureCMPModel()

    if  resume != None:
        if type(resume)==bool:
            path = './runs/latest.pth'
            if os.path.exists(path):
                dict = torch.load(path)
                model.load_state_dict(dict)
        else:
            path = resume
            dict = torch.load(path)
            model.load_state_dict(dict)

    model.optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
    model.loss_func = loss_fun()

    if torch.cuda.is_available(): 
        model = model.cuda()
    else:
        model = model.cpu()

    for i in range(EPOCHS):
        for index,(image, sub_imgs,label) in enumerate(dataloader):
            src   = model.img2Feature(image)
  
            while(len(sub_imgs)>0):

                BATCH =random.randint(MIN_BATCH,MAX_BATCH)

                BATCH = BATCH if len(sub_imgs)>BATCH else len(sub_imgs)

                tgt_batch = sub_imgs[0:BATCH]
                sub_imgs  = sub_imgs[BATCH:len(sub_imgs)]

                tgt = model.img2Feature(tgt_batch[0])

                for b in range(1,len(tgt_batch)):
                    tgt2 =  model.img2Feature(tgt_batch[b])
                    tgt = torch.concat([tgt,tgt2],0)
                
                loss = train_step(model,tgt,src,label)
            
            print('tick:{0} , batch:{1}  ,  loss: {2}'.format(i,index,loss))

            if ((i+1) %SAVE_CYCLE)==0:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                torch.save(model.state_dict(),'./runs/{0}_{1}.pth'.format(i,index))
                torch.save(model.state_dict(),'./runs/latest.pth')

if __name__=='__main__':
    train(    
        train_folder = './test/test',
        save_path = './runs/',
        EPOCHS=20000,
        MIN_BATCH = 1,
        MAX_BATCH = 32,
        SAVE_CYCLE = 2000,
        resume=True
        )
#  train_folder = '/home/zhizizhang/test/B00AMYTXM0/',