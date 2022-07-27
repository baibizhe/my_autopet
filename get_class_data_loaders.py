from monai.utils import first
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
from monai.data import DataLoader, Dataset,SmartCacheDataset
from monai.transforms import (
    EnsureTyped,
    Compose,
    LoadImaged,
    RandRotate90d,
    ScaleIntensityd,
    Resized, RandGaussianSmoothd, RandAdjustContrastd, RandGaussianNoised, RandShiftIntensityd,
)
import numpy as np
from monai.utils import first


def get_class_data_loaders(config):
    data_dir = os.path.join("data", "FDG-PET-CT-Lesions")
    all_patients = os.listdir(data_dir)
    all_patients = list(map(lambda x: os.path.join(data_dir, x), all_patients))
    all_patients_path = []
    labelPaths = []
    wealthy_p_count = 0

    imgPaths = []
    for i, patient in enumerate(all_patients):
        sub_patients = os.listdir(patient)
        for sub_patient in sub_patients:
            patient_dir = os.path.join(patient, sub_patient)
            imgPaths.append(os.path.join(patient_dir, "training_data_image.npy"))
            labelPaths.append(os.path.join(patient_dir, "training_data_marker.npy"))
            # print(patient_dir)

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(imgPaths, labelPaths)
    ]
    print(data_dicts)
    splitIndex = int(len(imgPaths) * 0.8)
    train_files, val_files = data_dicts[:], data_dicts[splitIndex:]
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        Resized(spatial_size=(128, 128, 128), keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        RandRotate90d(keys=["image"]),
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.15),
            sigma_y=(0.5, 1.15),
            sigma_z=(0.5, 1.15),
            prob=0.1,
        ),
        RandAdjustContrastd(keys=["image"], gamma=(0.5, 2.5), prob=0.1),
        RandGaussianNoised(
            keys=["image"],
            prob=0.1,
            mean=np.random.uniform(0, 0.5),
            std=np.random.uniform(0, 15),
        ),
        RandShiftIntensityd(
            keys=["image"], offsets=np.random.uniform(0, 0.3), prob=0.1
        ),
        EnsureTyped(keys=["image", "label"])
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        Resized(spatial_size=(128, 128, 128), keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "label"])
    ])

    train_ds = SmartCacheDataset(
        data=train_files, transform=train_transforms,
    )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)

    val_ds = SmartCacheDataset(
        data=val_files, transform=val_transforms)

    val_org_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image = (check_data["image"][0][0])
    label = check_data["label"][0][0]
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.title("image")
    plt.imshow(image[:, :, 50], cmap="gray")
    plt.savefig('data_loader.png')
    return train_loader, val_org_loader


if __name__ == '__main__':
    config = SimpleNamespace(patchshape=(96, 96, 96))
    get_class_data_loaders(config)
