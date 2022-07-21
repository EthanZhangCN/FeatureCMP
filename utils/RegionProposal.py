import cv2
from utils.rpn import RegionProposalNetwork
import numpy as np
import torch
import torch.nn as nn

def vgg_extractor():
    import torchvision.models.vgg as VGG
    vgg16 = VGG.vgg16(pretrained=True)
    features    = list(vgg16.features)[:30]
    extractor   = nn.Sequential(*features)
    extractor.eval()
    return extractor

def resnet50_extractor():
    import torchvision.models.resnet as ResNet
    resnet50 = ResNet.resnet50(pretrained=True)
    features    = list([resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool, resnet50.layer1, resnet50.layer2, resnet50.layer3])
    extractor   = nn.Sequential(*features)
    extractor.eval()
    return extractor

class RegionProposal(nn.Module):
    def __init__(
        self,
        backbone        = 'resnet50',
        ratios          = [0.5, 1, 2],
        anchor_scales   = [4, 16, 32],
        feat_stride     = 16,
        mode            = "predict", # or "training"
        min_size        = 100,
        debug_mod       = False
    ):
        super(RegionProposal,self).__init__()
        self.ratios         = ratios
        self.anchor_scales  = anchor_scales
        self.feat_stride    = feat_stride
        self.mode           = mode
        self.min_size       = min_size
        self.debug_mod      = debug_mod

        if backbone == 'vgg':
            self.in_channels        = 512
            self.rpn_state_dic_path = ''
            self.extractor          = vgg_extractor()
        elif backbone == 'resnet50':
            self.in_channels        = 1024
            self.rpn_state_dic_path = 'utils/resnet50_rpn.pth'
            self.extractor          = resnet50_extractor()

        self.rpn            =  RegionProposalNetwork(
                                    in_channels     = self.in_channels,
                                    mid_channels    = 512,
                                    ratios          = self.ratios,
                                    anchor_scales   = self.anchor_scales,
                                    feat_stride     = self.feat_stride,
                                    mode            = self.mode,
                                    min_size        = self.min_size
                                )

        self.rpn.load_state_dict(torch.load(self.rpn_state_dic_path))
        self.rpn.eval()

    def preprocess(self,image):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        self.img_size =torch.Size((image.shape[0],image.shape[1]))
        image = np.expand_dims(np.transpose(image, (2, 0, 1)), 0)
        # image = np.expand_dims(image, axis=0)
        # image = np.swapaxes(image,-1,1)
        return image

    def predict(self,image):
        if self.debug_mod:
            output_img = image.copy()

        image = self.preprocess(image)
        input_Tensor = torch.Tensor(image)
        input_Tensor = input_Tensor.to(self.device())
        base_feature = self.extractor.forward(input_Tensor)
        _, _, rois, roi_indices, _  = self.rpn.forward(base_feature, self.img_size, 1)

        if self.debug_mod:
            for i in range(len(rois[:, 0])):
                output_img = cv2.rectangle(output_img, (int(rois[i, 0]), int(rois[i, 1])), (int(rois[i, 2]),int(rois[i, 3])), (255, 255, 255),1)    
                # output_img = cv2.rectangle(output_img, (int(rois[i, 1]), int(rois[i, 0])), (int(rois[i, 3]),int(rois[i, 2])), (255, 255, 255),1)
            cv2.imwrite("./test/out.jpg", output_img)

        return rois

    def device(self):
        return next(self.parameters()).device

def resize_512(image):
    nw = 512 if image.shape[1] < image.shape[0] else int(512 * image.shape[1] / float(image.shape[0]))
    nh = 512 if image.shape[0] < image.shape[1] else int(512 * image.shape[0] / float(image.shape[1]))
    image = cv2.resize(image,(nw,nh))
    return image

if __name__ == '__main__':
    test_img_path = "./test/1.jpg"
    image = cv2.imread(test_img_path)
    image = cv2.resize(image,(512,512))
    image = resize_512(image)
    rpn = RegionProposal(min_size=64,debug_mod=True)
    rois =  rpn.predict(image)
    print(rois.size())
