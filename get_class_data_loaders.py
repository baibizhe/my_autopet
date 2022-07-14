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
    SpatialPadd,
    ScaleIntensityd,
    Resized,
)
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd, RandCropByLabelClassesd
from monai.data import  DataLoader, Dataset, decollate_batch
from  types import  SimpleNamespace
import matplotlib.pyplot as plt
import os
import numpy as np
def get_class_data_loaders(config):
    data_dir = os.path.join("data","FDG-PET-CT-Lesions")
    all_patients = os.listdir(data_dir)
    all_patients =list(map(lambda x: os.path.join(data_dir,x),all_patients))
    all_patients_path =[]
    labelPaths = []
    wealthy_p_count = 0
    
    imgPaths = []
    for i,patient in enumerate(all_patients):
        sub_patients = os.listdir(patient)
        for sub_patient in sub_patients:
            patient_dir = os.path.join(patient,sub_patient)
            imgPaths.append(os.path.join(patient_dir,"training_data_image.npy"))
            labelPaths.append(os.path.join(patient_dir,"training_data_marker.npy"))
            # print(patient_dir)


    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(imgPaths, labelPaths)
    ]
    print(data_dicts)
    splitIndex = int(len(imgPaths) * 0.8)
    train_files, val_files = data_dicts[:splitIndex], data_dicts[splitIndex:]
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
                ScaleIntensityd(keys=["image"]), 
                                AddChanneld(keys=["image", "label"]), 
                                Resized(spatial_size=(400, 200, 200),keys=["image"]), 
                                RandRotate90d(keys=["image", "label"]), 
                                EnsureTyped(keys=["image", "label"])])
    val_transforms = Compose([ 
        LoadImaged(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]), 
                              AddChanneld(keys=["image", "label"]), 
                              Resized(spatial_size=(400, 200, 200),keys=["image"]), 
                              EnsureTyped(keys=["image", "label"])])

    train_ds = Dataset(
    data=train_files, transform=train_transforms,
     )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=3)

    val_ds = Dataset(
        data=val_files, transform=val_transforms)
    
    val_org_loader = DataLoader(val_ds, batch_size=1,shuffle=False, num_workers=3)
    
    check_ds = Dataset(data=val_files, transform=val_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image = (check_data["image"][0][0])
    label = check_data["label"][0][0]
    print(f"image shape: {image.shape}, label shape: {label.shape}")
    # plot the slice [:, :, 80]
    plt.figure("check", (12, 6))
    plt.title("image")
    plt.imshow(image[:, :, 80], cmap="gray")
    plt.savefig('data_loader.png')
    return   train_loader,val_ds

if __name__ == '__main__':
    config = SimpleNamespace(patchshape=(96,96,96))
    get_class_data_loaders(config)

