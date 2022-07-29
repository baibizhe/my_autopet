import argparse
import os
import random
from types import SimpleNamespace

from skimage.util import montage
from torch.nn import init

import matplotlib.pyplot as plt
import monai.networks.nets
import numpy as np
import torch
import wandb
from monai.inferers import sliding_window_inference
from timm.utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from get_seg_data_loaders import get_data_loaders
from tensor_board import Tensorboard
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from torch.optim import  AdamW
from val_script import  dice_score



def train_epoch(model, train_loader, optimizer, device, epoch, trainepochs, loss_fn1, loss_fn2):
    model.train()
    losses = AverageMeter()
    scaler = GradScaler()
    dice_coefficients = AverageMeter()

    for batch_idx, batch in enumerate(train_loader):
        data, target = batch["image"].to(device), batch["label"].to(device).long()
        if target.sum() == 0 :
            continue
        optimizer.zero_grad()
        with autocast():
            output = model(data)

            # loss = loss_fn1(output.unsqueeze(2), target)  # * 1000
            loss = loss_fn2(output, target)
            output = torch.argmax(output, 1)

        dice_coefficients.update(dice_score(output.detach().cpu().numpy(), target.cpu().numpy()), len(batch))
        losses.update(loss.item(), data.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.step()
    return losses.avg, dice_coefficients.avg


def val_epoch_metrics(model, val_org_loader, post_transforms, patchshape):
    model.eval()
    dice_coefficients = AverageMeter()
    out_up , label_up , input_up =None,None,None
    currnt_sum_seg_not_zero = 0
    with torch.no_grad():
        for idx, val_data in enumerate(val_org_loader):
            val_inputs = val_data["image"].cuda()
            roi_size = patchshape
            sw_batch_size = 1
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            val_data["pred"] = torch.argmax(val_data["pred"], dim=1, keepdim=True)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_coefficients.update(
                dice_score(val_outputs[0].numpy().astype(int), val_labels[0].numpy().astype(int)), 1)
            if val_labels[0].sum().sum() > currnt_sum_seg_not_zero:
                currnt_sum_seg_not_zero = val_labels[0].sum().sum() #返回最大值
                out_up = val_outputs
                label_up = val_labels
                input_up = val_inputs


    return  dice_coefficients.avg,out_up[0][0],label_up[0][0],input_up[0][0]

def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm3d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def get_model(device, config):
    outputChannel = 14

    model = monai.networks.nets.SwinUNETR(img_size=config.patchshape, in_channels=2, out_channels=2,
                                          feature_size=48).cuda()
    # model = monai.networks.nets.VNet(in_channels=2,out_channels=2,dropout_prob=0.1)
    model.load_from(torch.load("model_swinvit.pt"))
    # init_weights(model, init_type="kaiming")

    if config.resumePath != '':
        print("loading model from {}".format(config.resumePath))
        model.load_state_dict(torch.load(config.resumePath), strict=True)
    else:
        print("new training")
        # model._init_weights()
    model.to(device)
    return model


def main():
    # seed_everything()
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--batch-size", type=int, default=1

        )
    arg("--epochs", type=int, default=1000)
    arg("--lr", type=float, default=0.00002
        )
    arg("--workers", type=int, default=6)
    arg("--model", type=str, default="ResUnet3D")
    arg("--patchshape", type=tuple, default=(96, 96, 96))
    #     arg("--test_mode", type=str2bool, default="false",choices=[True,False])
    arg("--optimizer", type=str, default="AdamW")

    arg("--taskname", type=str, default="pretrain+96patch")

    arg("--pretrain-path", type=str, default='model_bestValRMSE.pt')
    arg("--resumePath", type=str, default='')
    arg("--use-wandb", action="store_true",)

    arg(
        "--device-ids",
        type=str,
        default="0",
        help="For example 0,1 to run on two GPUs",
    )
    args = parser.parse_args()
    print(args)
    os.environ['WANDB_API_KEY'] = "55a895793519c48a6e64054c9b396629d3e41d10"

    if args.use_wandb:
        mywandb = Tensorboard(config=args)
    if not os.path.exists("outPutImages"):
        try:
            os.makedirs("outPutImages")
        except:
            pass
    baseRoot = os.path.join("expOutput", args.taskname)
    if not os.path.exists(baseRoot):
        try:
            os.makedirs(baseRoot)
        except:
            pass
    device = torch.device("cuda:%d" % 0)
    # post_transforms, train_loader, val_loader = get_data_loaders(args)
    train_loader, val_org_loader, post_transforms = get_data_loaders(args)
    model = get_model(device=device, config=args)

    optimizer = eval(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95, last_epoch=-1)
    warmup_epochs = 2
    T_mult = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                     T_mult=T_mult)
    # will restart at 5+5*2=15，15+10*2=35，35+20 * 2=75

    epochs = args.epochs
    CELossFun = torch.nn.CrossEntropyLoss()
    DiceLossFun = monai.losses.DiceCELoss(to_onehot_y=True,
                                          softmax=True)


    with trange(epochs) as t:
        for epoch in t:
            trainLosses, trainDiceCoefficients, validLosses, validDiceCoefficients = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            validLossEpoch, validDiceEpoch, valid_dice = 0, 0, 0
            # if epoch == 3:
            #     for param in model.swinViT.parameters():
            #         param.requires_grad = True
            t.set_description('Epoch %i' % epoch)
            trainLossEpoch, trainDiceEpoch = train_epoch(model=model,
                                                         train_loader=train_loader,
                                                         optimizer=optimizer,
                                                         device=device,
                                                         epoch=epoch,
                                                         loss_fn1=CELossFun,
                                                         loss_fn2=DiceLossFun,
                                                         trainepochs=epochs)
            # trainLossEpoch, trainDiceEpoch = 0,0

            if epoch % 2 == 0:
                validDiceEpoch,out_up,label,image_CTres  = val_epoch_metrics(model=model,
                                                                   val_org_loader=val_org_loader,
                                                                   post_transforms=post_transforms,
                                                                   patchshape=args.patchshape)
                non_zero_idx = torch.where(label.sum((1, 2)) > 0)
                image_CTres = image_CTres[non_zero_idx].cpu().numpy()
                label = label[non_zero_idx].cpu().numpy()
                out_up = out_up[non_zero_idx].cpu().numpy()
                # image_CTres += label
                # image_SUV += label

                fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=200)
                ax1.imshow(montage(image_CTres), cmap='bone')
                plt.savefig(os.path.join("outPutImages",'image_CTres_epoch{}.png'.format(epoch)))
                fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=200)
                ax1.imshow(montage(out_up), cmap='bone')
                plt.savefig(os.path.join("outPutImages",'image_Pred_epoch{}.png'.format(epoch)))

                fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=200)
                ax1.imshow(montage(label), cmap='bone')
                plt.savefig(os.path.join("outPutImages",'image_label_epoch{}.png'.format(epoch)))


            trainLosses.update(trainLossEpoch)
            trainDiceCoefficients.update(trainDiceEpoch)
            validDiceCoefficients.update(validDiceEpoch)
            lr = optimizer.param_groups[0]["lr"]


            info_dict = {"Train loss ": trainLossEpoch,
                           "Train Dice resized": trainDiceEpoch,
                           "Valid full dice ": validDiceEpoch,
                           "lr": lr,

                           }

            t.set_postfix(info_dict)

            if validDiceEpoch > valid_dice:
                torch.save(model.state_dict(), os.path.join(baseRoot, args.taskname + str(epoch) + ".pth"))
                print("模型保存")
                valid_dice = validDiceEpoch

            scheduler.step()




if __name__ == "__main__":
    main()