import argparse
import os.path

import torch.nn
from sklearn.model_selection import KFold
from timm.utils import AverageMeter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm,trange
from TrainConfig import config

from get_training_components import get_all_patients_dir_paths, get_model, get_lr_scheduler, get_loss_function, \
    get_augmentations, get_wandb, get_optimizer, get_dataloaders
from tensor_board import Tensorboard

from utils import compute_metrics




def cal_metrics(output, label,config):
    # return 1,1,1
    output = output.cpu().detach()
    label = label.cpu().detach()
    output = torch.argmax(output,1).numpy()
    label = torch.argmax(label,1).numpy()
    return compute_metrics(output,label,config)



def train_one_epoch(model, train_loader, optimizer, loss_function,scaler):
    dice_sc,losses = AverageMeter(),AverageMeter()
    inner_bar = tqdm(train_loader, desc = 'training', leave = False)
    for idx,(image,label) in enumerate(train_loader):
        # print("cur itr {} total {} itr".format(idx,len(train_loader)))
        optimizer.zero_grad()
        image,label =image.to(config.device) ,label.float().squeeze(1).to(config.device)

        with autocast():
            output = model(image)
            loss = loss_function(output,label)
            losses.update(loss.item(),image.shape[0])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        dice_sc_avg, false_pos_vol_avg, false_neg_vol_avg = cal_metrics(output,label,config)
        dice_sc.update(dice_sc_avg)
        inner_bar.update(1)
        # metrics_info = {"dice_sc":dice_sc.avg,
        #                 "losses":losses.avg}
        # print(metrics_info)
    metrics_info = {"train dice_sc": dice_sc.avg,
                    "train losses": losses.avg}
    return  metrics_info


def valid_one_epoch(model, valid_loader, loss_function):
    dice_sc, losses = AverageMeter(), AverageMeter()

    for idx, (image, label) in enumerate(valid_loader):
        image, label = image.to(config.device), label.float().squeeze(1).to(config.device)
        with torch.no_grad():
            with autocast():
                output = model(image)
                loss = loss_function(output, label)
                losses.update(loss.item(), image.shape[0])
            dice_sc_avg, false_pos_vol_avg, false_neg_vol_avg = cal_metrics(output, label, config)
            dice_sc.update(dice_sc_avg)
        # metrics_info = {"dice_sc":dice_sc.avg,
        #                 "false_pos_vol":false_pos_vol.avg,
        #                 "false_neg_vol":false_neg_vol.avg,
        #                 "losses":losses.avg}
        # print(metrics_info)
    metrics_info = {"valid dice_sc": dice_sc.avg,
                    "valid losses": losses.avg}
    return metrics_info


def main(cmd_line_var):
    config.update(vars(cmd_line_var))
    print(config)

    scaler = GradScaler()

    all_patients_path = get_all_patients_dir_paths(config.dataPath)[0:100]

    if config.use_wandb:
        mywandb = get_wandb(config)
    kf = KFold(n_splits=5)
    print("using device {}".format(str(config.device)))
    for train_index, test_index in kf.split(all_patients_path):
        model = get_model(config).to(config.device)
        train_augmentations, valid_augmentations = get_augmentations(config)
        optimizer_params = {"lr": config.learning_rate, "weight_decay": config.weight_decay}
        optimizer = get_optimizer(model=model,
                                  optimizer_name=config.optimizer_name,
                                  optimizer_params=optimizer_params)
        lr_scheduler = get_lr_scheduler(config, optimizer)
        loss_function = get_loss_function(config)
        patients_path_train, patients_path_valid = all_patients_path[train_index], all_patients_path[test_index]
        print("training length {} valid length {}".format(len(patients_path_train),len(patients_path_valid)))
        with tqdm(total=config.epochs, bar_format=" training : {postfix[0]}  valid:{postfix[1]}",
                  postfix=[dict(), dict(value=0)]) as t:
            for i in enumerate(tqdm(range(config.epochs))):
                train_loader, valid_loader = get_dataloaders(patients_path_train,
                                                             patients_path_valid,
                                                             train_augmentations,
                                                             valid_augmentations,
                                                             config,
                                                             )

                train_metrics = train_one_epoch(model=model,
                                                  train_loader= train_loader,
                                                  optimizer=optimizer,
                                                loss_function=loss_function,
                                                scaler=scaler)
                mywandb.upload_wandb_info(info_dict=train_metrics)
                # print("train_metrics",train_metrics)
                valid_metrics = valid_one_epoch(model=model,
                                              valid_loader= valid_loader,
                                                loss_function=loss_function
                                                )
                mywandb.upload_wandb_info(info_dict=valid_metrics)

                # print("valid_metrics:",valid_metrics)
                t.postfix[0] = train_metrics

                t.postfix[1] = valid_metrics
                t.update()

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='autoPET Semantic Segmentation')
    # network architectures
    parser.add_argument("-a", "--model", default='SwinUNETR', type=str,
                        choices=["SwinUNETR","VNET","UNETR"],
                        help="select the architecture in use")

    parser.add_argument("--dataPath",default=os.path.join('data',"resize_128_data"), help="wandb key for docker")
    parser.add_argument("--batch_size",default=2, type=int)

    cmd_line_var = parser.parse_args()
    main(cmd_line_var)