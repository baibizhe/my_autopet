import numpy as np
import torch.nn
import torch.optim
from monai import transforms
from monai.networks.nets import SwinUNETR, VNet, UNETR
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
import os
from TrainConfig import config
from dataset import CustomTrainImageDataset
# def get_all_patients_dir_paths(dataPath:str)->np.ndarray:
#     patients = list(map(lambda x: os.path.join(dataPath, x), os.listdir(dataPath)))
#     patients_paths = []
#     for patient in patients:
#         sub_dir = os.listdir(patient)
#         sub_dir_path = os.path.join(patient, sub_dir[0])
#         patients_paths.append(sub_dir_path)
#     return  np.array(patients_paths)
from tensor_board import Tensorboard


def get_all_patients_dir_paths(dataPath:str)->np.ndarray:
    patients_paths = list(map(lambda x: os.path.join(dataPath, x), os.listdir(dataPath)))
    return  np.array(patients_paths)


def get_model(config)->torch.nn.Module:
    input_channel = 2
    output_channel = 2
    if config.model == "SwinUNETR":
        model = SwinUNETR(img_size=config.target_size,
                          in_channels=input_channel,
                          out_channels=output_channel,
                          feature_size=12,
                          depths=(1, 1, 1, 1),
                          norm_name="batch",
                          drop_rate=config.drop_out_rate,
                          )
        print(model)
        return  model

    if config.model == "VNET":
        model = VNet(in_channels=input_channel,
                     out_channels=output_channel,
                     dropout_prob=config.drop_out_rate,
                     )
        print(model)

        return  model


    if config.model == "UNETR":

        model = UNETR(img_size=config.target_size,
                      in_channels=input_channel,
                      out_channels=output_channel,
                      pos_embed="perceptron",
                      )
        print(model)

        return  model


def get_lr_scheduler(config,optimizer) :
    if config.lr_sheduler != "ExponentialLR" and config.lr_sheduler != "CosineAnnealingWarmRestarts":
        raise NotImplementedError
    lr_sheduler = ExponentialLR(optimizer=optimizer,
                               gamma=0.99,
                                )
    if config.lr_sheduler == "ExponentialLR":
        return lr_sheduler
    lr_sheduler = CosineAnnealingWarmRestarts(optimizer=optimizer,
                                              T_0=1,
                                              last_epoch=config.epochs)
    if config.lr_sheduler == "CosineAnnealingWarmRestarts":
        return lr_sheduler


def get_loss_function(config):
    return  CrossEntropyLoss()

    return  BCEWithLogitsLoss()


def get_augmentations(config):
    train_transform = transforms.Compose([
            transforms.Resized(keys=["image", "label"],spatial_size=config.target_size),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # transforms.NormalizeIntensityd(
            #     keys="image", nonzero=True, channel_wise=True
            # ),
            # transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            # transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            # transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose([
        # transforms.Resize(spatial_size=config.target_size),
        # transforms.NormalizeIntensityd(
        #     keys="image", nonzero=True, channel_wise=True
        # ),
        # transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return  train_transform,val_transform


def get_wandb(config) -> Tensorboard:
    # return
    tensorboard = Tensorboard(config=config)
    return tensorboard



def get_optimizer(model, optimizer_name, optimizer_params):
    _OPTIMIZERS_REGISTRY = {
        "asgd": torch.optim.ASGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    optimizer=_OPTIMIZERS_REGISTRY[config.optimizer_name](model.parameters(),
                                                              lr=config.learning_rate,
                                                              weight_decay=config.weight_decay)
    return optimizer


def get_dataloaders(patients_path_train, patients_path_valid, train_augmentations, valid_augmentations, config):
    train_dataset = CustomTrainImageDataset(allImagePath=patients_path_train,
                                            augmentation=train_augmentations)

    valid_dataset = CustomTrainImageDataset(allImagePath=patients_path_valid,
                                            augmentation=valid_augmentations)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    return train_loader,valid_loader