import torch
from torchsummary import summary

from monai.networks.nets import SwinUNETR,UNETR,VNet
import os




def main():
    roi = (64, 64, 64)
    model = SwinUNETR(
        img_size=roi,
        in_channels=2,
        out_channels=2,
        feature_size=12,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
    ).cuda()
    x = torch.rand(size=(2,2,64,64,64)).cuda()
    model(x)
    # summary(model, (2, 256, 256,256))
    model = UNETR(
        img_size=roi,
        in_channels=4,
        out_channels=2,
        pos_embed="perceptron",
        # pos_embed="conv",#perceptron
    ).cuda()
    x = torch.rand(size=(1,4,128,128,128)).cuda()
    model(x)
    # summary(model, (4, 128, 128,128))

    model=VNet(in_channels=4,
               out_channels=2).cuda()
    summary(model, (4, 128, 128,128))

if __name__ == '__main__':
    main()