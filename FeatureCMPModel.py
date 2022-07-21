from turtle import forward
import torch.nn as nn

import torch
import math
from torch.autograd import Variable
import cv2
import numpy as np
import torchvision.models.vgg as VGG


class extractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vgg16 = VGG.vgg16(pretrained=False)
        features    = list(vgg16.features)[:31]
        self.features   = nn.Sequential(*features)
        self.maxpool = nn.MaxPool2d((7,7))
        dic_path = 'utils/vgg16_from_tf_notop.pth'
        state_dict =torch.load(dic_path)
        self.load_state_dict(state_dict)
        self.eval()

    def forward(self,x):
        x = self.features(x)
        return x
import pdb

class FeatureCMPModel(nn.Module):
    def __init__(self) -> None:
        super(FeatureCMPModel,self).__init__()
        self.transformer_model =nn.Transformer(nhead=16, num_encoder_layers=6)
        self.Feature_model = extractor()
        self.conv1d = nn.Conv1d(in_channels=49,out_channels=19,kernel_size=1)

        self.conv1d_2 = nn.Conv1d(in_channels=512,out_channels=32,kernel_size=7)
        self.conv1d_3 = nn.Conv1d(in_channels=32 ,out_channels=16,kernel_size=7)
        self.conv1d_4 = nn.Conv1d(in_channels=16 ,out_channels=1 ,kernel_size=7)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self,src_feat,tgt_feat):
        src_feat.to(self.device())
        tgt_feat.to(self.device())
        out = self.transformer_model.forward(src_feat, tgt_feat)

        out = self.conv1d(out)
        out = torch.transpose(out,1,2)
        out = self.conv1d_2(out)
        out = self.relu(out)
        out = self.conv1d_3(out)
        out = self.relu(out)
        out = self.conv1d_4(out)

        out = torch.flatten(out,1)

        out = self.sigmoid(out)

        return out

    def img2Feature(self,image):
        image = self.preprocess(image)
        input_Tensor = torch.Tensor(image)
        input_Tensor = input_Tensor.to(self.device())
        feat = self.Feature_model.forward(input_Tensor)
        feat = torch.flatten(feat,-2)
        feat = torch.transpose(feat,1,2)
        return feat.detach()

    def preprocess(self,image):
        image = cv2.resize(image,(224,224))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        self.img_size =torch.Size((image.shape[0],image.shape[1]))
        image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        return image
    
    def device(self):
        return next(self.parameters()).device
    
    

if __name__=="__main__":
    test_img_path = "./test/2.jpg"
    image = cv2.imread(test_img_path)
    m = FeatureCMPModel()
    src  = m.img2Feature(image)
    src =torch.concat([src,src],dim=0)
    src =torch.concat([src,src],dim=0)

    # TTT = torch.rand([1,49,512])
    r =  m.forward(src,src)
    print(r.size())
    exit()

# from buz.nets.rpn import RegionProposalNetwork

# # def resnet50_extractor():
# #     import torchvision.models.resnet as ResNet
# #     resnet50 = ResNet.resnet50(pretrained=True)
# #     features    = list([resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.layer3])
# #     extractor   = nn.Sequential(*features)
# #     extractor.eval()
# #     return extractor

# import torchvision.models.vgg as VGG
# class extractor(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         vgg16 = VGG.vgg16(pretrained=False)
#         features    = list(vgg16.features)[:31]
#         self.features   = nn.Sequential(*features)
#         self.maxpool = nn.MaxPool2d((7,7))
#         dic_path = 'buz/model_data/vgg16_from_tf_notop.pth'
#         state_dict =torch.load(dic_path)
#         self.load_state_dict(state_dict)
#         self.eval()

#     def forward(self,x):
#         x = self.features(x)
#         # x = self.maxpool(x)
#         # x = torch.flatten(x, 1)
#         return x



# def rpn_net(
#         self,        
#         backbone        = 'resnet50',
#         ratios          = [0.5, 1, 2],
#         anchor_scales   = [4, 16, 32],
#         feat_stride     = 16,
#         mode            = "predict", # or "training"
#         min_size        = 100,
#         debug_mod       = False):
    
#     self.ratios         = ratios
#     self.anchor_scales  = anchor_scales
#     self.feat_stride    = feat_stride
#     self.mode           = mode
#     self.min_size       = min_size
#     self.debug_mod      = debug_mod
#     self.in_channels = 1024
#     self.rpn =  RegionProposalNetwork(
#                                     in_channels     = self.in_channels,
#                                     mid_channels    = 512,
#                                     ratios          = ratios,
#                                     anchor_scales   = anchor_scales,
#                                     feat_stride     = feat_stride,
#                                     mode            = mode,
#                                     min_size        = min_size
#                                 )
#     self.rpn.load_state_dict(torch.load(self.rpn_state_dic_path))
#     self.rpn.eval()

#     return self.rpn


# class FeatureProposalNetwork(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.extractor = extractor()
#         self.transformer = nn.Transformer(nhead=16, num_encoder_layers=6,num_decoder_layers= 6 )
#         self.rpn            = rpn_net()

        
#     def forward(self,x):
#         x = self.extractor.forward(x) # 1 * 512 * 7 * 7



