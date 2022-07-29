import argparse
import os
import random
import time
from types import SimpleNamespace
from torch.nn import init
import torch.distributed as dist
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
from distri_seg_loader import get_distributed_data_loaders
from tensor_board import Tensorboard
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from torch.optim import  AdamW
from val_script import  dice_score
import torch.multiprocessing as mp
import torch.utils.data.distributed
from utils import  distributed_all_gather
import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
def train_epoch(model, train_loader, optimizer, args,scaler, loss_fn1, loss_fn2):
    model.train()
    losses = AverageMeter()
    scaler = GradScaler()
    dice_coefficients = AverageMeter()
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data, target = batch["image"].cuda(args.rank), batch["label"].cuda(args.rank).long()
        for param in model.parameters(): param.grad = None
        if target.sum() == 0 :
            continue
        print(target.sum(),)

        # optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = loss_fn1(output.unsqueeze(2), target)  # * 1000
            output = torch.argmax(output, 1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        dice_coefficient = torch.tensor(dice_score(output.detach().cpu().numpy(), target.cpu().numpy()),dtype=torch.float32)
        print(args.distributed)
        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=batch_idx < train_loader.sampler.valid_length)
            losses.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
            diceList = None
            torch.distributed.all_gather(tensor_list = [dice_coefficient],
                                                tensor=diceList)
            print(diceList)



        optimizer.step()
    for param in model.parameters(): param.grad = None
    return losses.avg, dice_coefficients.avg


def val_epoch_metrics(model, val_org_loader, post_transforms, patchshape):
    model.eval()
    dice_coefficients = AverageMeter()
    out_up , label_up =None,None
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
            if idx == 1:
                out_up = val_outputs[0].cpu()[0].numpy()[:, :, 60]
                label_up = val_labels[0].cpu()[0].numpy()[:, :, 60]


    return  dice_coefficients.avg,out_up,label_up

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
    # model = monai.networks.nets.VNet(in_channels=2,out_channels=2)
    model.load_from(torch.load("model_swinvit.pt"))
    # init_weights(model, init_type="kaiming")

    if config.resumePath != '':
        print("loading model from {}".format(config.resumePath))
        model.load_state_dict(torch.load(config.resumePath), strict=True)
    else:
        print("new training")
        # model._init_weights()
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
    arg('--rank', default=0, type=int, help='node rank for distributed training')
    arg('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
    arg('--dist-backend', default='nccl', type=str, help='distributed backend')
    arg(
        "--device-ids",
        type=str,
        default="0",
        help="For example 0,1 to run on two GPUs",
    )
    arg('--norm_name', default='instance', type=str, help='normalization name')
    arg('--world_size', default=1, type=int, help='number of nodes for distributed training')
    arg('--distributed', action='store_true', help='start distributed training')
    args = parser.parse_args()
    print(args)
    os.environ['WANDB_API_KEY'] = "55a895793519c48a6e64054c9b396629d3e41d10"
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu,args):
    print("记得backend 改成ncll  startmethod改成fork")
    if args.distributed:
        torch.multiprocessing.set_start_method('spawn', force=True)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False
    print(args.rank, ' gpu', args.gpu)
    model = get_model(device="cuda", config=args).cuda(args.gpu)
    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          )
    print("finish initial parallell")
        # torch.multiprocessing.set_start_method('fork', force=True)

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
    train_loader, val_org_loader, post_transforms = get_distributed_data_loaders(args)
    optimizer = eval(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.95, last_epoch=-1)
    warmup_epochs = 3
    T_mult = 2
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                     T_mult=T_mult)
    # will restart at 5+5*2=15，15+10*2=35，35+20 * 2=75
    epochs = args.epochs
    CELossFun = torch.nn.CrossEntropyLoss()
    DiceLossFun = monai.losses.DiceCELoss(to_onehot_y=True,
                                          softmax=True)
    scaler = GradScaler()
    with trange(epochs) as t:
        for epoch in t:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
                torch.distributed.barrier()
            print(args.rank, time.ctime(), 'Epoch:', epoch)
            trainLosses, trainDiceCoefficients, validLosses, validDiceCoefficients = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            validLossEpoch, validDiceEpoch, valid_dice = 0, 0, 0
            # if epoch == 3:
            #     for param in model.swinViT.parameters():
            #         param.requires_grad = True
            t.set_description('Epoch %i' % epoch)
            trainLossEpoch, trainDiceEpoch = train_epoch(model=model,
                                                         train_loader=train_loader,
                                                         optimizer=optimizer,
                                                         args=args,
                                                         loss_fn1=CELossFun,
                                                         loss_fn2=DiceLossFun,
                                                         scaler=scaler
                                                        )
            # trainLossEpoch, trainDiceEpoch = 0,0

            if epoch % 2 == 0:
                validDiceEpoch, out_up, label_up = val_epoch_metrics(model=model,
                                                                     val_org_loader=val_org_loader,
                                                                     post_transforms=post_transforms,
                                                                     patchshape=args.patchshape)

            trainLosses.update(trainLossEpoch)
            trainDiceCoefficients.update(trainDiceEpoch)
            validDiceCoefficients.update(validDiceEpoch)
            lr = optimizer.param_groups[0]["lr"]

            info_dict = {"Train loss ": trainLossEpoch,
                         "Train Dice resized": trainDiceEpoch,
                         "Valid full dice ": validDiceEpoch,
                         "lr": lr,

                         }
            # if args.use_wandb:
            #     mywandb.upload_wandb_info(info_dict=info_dict)
            #     plt.figure("check", (12, 6))
            #     plt.subplot(1, 2, 1)
            #     plt.title("image")
            #     plt.imshow(out_up)
            #     plt.subplot(1, 2, 2)
            #     plt.title("label")
            #     plt.imshow(label_up)
            #
            #     mywandb.tensor_board.log({"data": wandb.Image(plt)})
            # else:
            #     plt.figure("check", (12, 6))
            #     plt.subplot(1, 2, 1)
            #     plt.title("image")
            #     plt.imshow(out_up)
            #     plt.subplot(1, 2, 2)
            #     plt.title("label")
            #     plt.imshow(label_up)
            #
            #     plt.savefig(os.path.join("outPutImages",'output{}.png'.format(epoch)))
            t.set_postfix(info_dict)

            if validDiceEpoch > valid_dice:
                torch.save(model.state_dict(), os.path.join(baseRoot, args.taskname + str(epoch) + ".pth"))
                print("模型保存")
                valid_dice = validDiceEpoch

            scheduler.step()


if __name__ == "__main__":
    main()