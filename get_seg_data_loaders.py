import os
from monai.utils import first, set_determinism
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    EnsureTyped,
    Invertd, RandFlipd, RandShiftIntensityd,
        AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd
)
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd, RandCropByLabelClassesd
from monai.data import  DataLoader, Dataset, decollate_batch
from  types import  SimpleNamespace
import matplotlib.pyplot as plt
import os
import numpy as np
def get_data_loaders(config):
    dataDirPath = "data/FLARE22_LabeledCase50-20220324T003930Z-001"
    # dataDirPath = "data/FLARE22_LabeledCase50"

    imgPaths = list(
        map(lambda x: os.path.join(dataDirPath, "images", x), os.listdir(os.path.join(dataDirPath, "images"))))
    labelPath = list(
        map(lambda x: os.path.join(dataDirPath, "labels", x), os.listdir(os.path.join(dataDirPath, "labels"))))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(imgPaths, labelPath)
    ]
    splitIndex = int(len(imgPaths) * 0.9)
    train_files, val_files = data_dicts[:splitIndex], data_dicts[splitIndex:]
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             Orientationd(keys=["image", "label"], axcodes="RAS"),

#             ResizeWithPadOrCropd(keys=["image", "label"],spatial_size=(512,512,config.patchshape[2])),
#             RandCropByLabelClassesd(
#                 keys=["image", "label"],
#             label_key="label",
#                 spatial_size=config.patchshape,
#                 num_classes=14,
#                     num_samples=4,
#                 ratios=[1 for i in range(14)],

#             ),
#             ScaleIntensityRanged(
#                 keys=["image"], a_min=-57, a_max=164,
#                 b_min=0.0, b_max=1.0, clip=True,
#             ),
#             EnsureTyped(keys=["image", "label"]),
#         ]
#     )

    train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"],spatial_size=(-1,-1,96)),
            # RandCropByLabelClassesd(
            #     keys=["image", "label"],
            # label_key="label",
            #     spatial_size=config.patchshape,
            #     num_classes=14,
            #         num_samples=3,
            #     # ratios=[1 for i in range(14)],
            # ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=3,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)
#     val_org_transforms = Compose(
#         [
#             LoadImaged(keys=["image", "label"]),
#             EnsureChannelFirstd(keys=["image", "label"]),
#             ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(512, 512,config.patchshape[2])),

#             Orientationd(keys=["image", "label"], axcodes="RAS"),
#             EnsureTyped(keys=["image", "label"]),
#         ]
#     )

    val_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"],spatial_size=(512,512,96)),
        
        ToTensord(keys=["image", "label"]),
    ]
    ) 

    train_ds = Dataset(
        data=train_files, transform=train_transforms,
         )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=3)

    val_org_ds = Dataset(
        data=val_files, transform=val_org_transforms)

    post_transforms = Compose([
        EnsureTyped(keys=["pred","label"]),
        Invertd(
            keys=["pred","label"],
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys=["pred_meta_dict","label_meta_dict"],
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        # AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        # AsDiscreted(keys="label", to_onehot=2),
    ])
    val_org_loader = DataLoader(val_org_ds, batch_size=1,shuffle=False, num_workers=3)

    check_ds = Dataset(data=val_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image, label = (check_data["image"][0][0], check_data["label"][0][0])
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(image[:, :, 80], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[:, :, 80])
    plt.savefig('data_loader.png')
    return   train_loader,val_org_loader,post_transforms

if __name__ == '__main__':
    config = SimpleNamespace(patchshape=(96,96,96))
    get_data_loaders(config)

