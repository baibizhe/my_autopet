import os

import torch
from monai.utils import first, set_determinism
from skimage.util import montage
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
HistogramNormalized,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd, NormalizeIntensityd, RandScaleIntensityd
)
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropd, RandCropByLabelClassesd, RandSpatialCropd, \
    RandSpatialCropSamplesd
from monai.data import DataLoader, Dataset, decollate_batch, SmartCacheDataset
from  types import  SimpleNamespace
import matplotlib.pyplot as plt
import os
import numpy as np
import math
class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
def threshold_at_one(x):
    # threshold at 1
    return x >= 1
def get_distributed_data_loaders(args):
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

            labelPaths.append(os.path.join(patient_dir, "SEG.nii.gz"))
            # print(patient_dir)

    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(imgPaths, labelPaths)
    ]
    print(data_dicts)
    splitIndex = int(len(imgPaths) * 0.9)


    train_files, val_files = data_dicts[:splitIndex], data_dicts[splitIndex:]
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=[ "label"]),
            Orientationd(keys=["label"], axcodes="SRA"),
            CropForegroundd(keys=["image", "label"], source_key="image",select_fn=threshold_at_one),
            # HistogramNormalized(keys=["image"]),

            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=(96,96,96),
                num_samples=1,
                random_size=False,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )



    val_org_transforms =     Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=[ "label"]),
            Orientationd(keys=["label"], axcodes="SRA"),
            CropForegroundd(keys=["image", "label"], source_key="image",select_fn=threshold_at_one),
            HistogramNormalized(keys=["image"]),

            RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=(96,96,96),
                random_center=True,
                num_samples=1,
                random_size=False,
            ),

            EnsureTyped(keys=["image", "label"]),
        ]
    )

    train_ds = Dataset(
        data=train_files, transform=train_transforms,
         )
    # train_ds = Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = DataLoader(train_ds,
                              batch_size=1,
                              shuffle=(train_sampler is None),
                              sampler = train_sampler,
                              num_workers=0,
                                pin_memory = True,
                              )

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
    val_sampler = Sampler(val_org_ds, shuffle=False) if args.distributed else None
    val_org_loader = DataLoader(val_org_ds, batch_size=1,shuffle=False, num_workers=0,sampler=val_sampler,pin_memory=True)

    check_ds = Dataset(data=train_files, transform=train_transforms)
    check_loader = DataLoader(check_ds, batch_size=1)
    check_data = first(check_loader)
    image_CTres,image_SUV,label = (check_data["image"][0][0],check_data["image"][0][1], check_data["label"][0][0])
    print(f"image shape: {image_CTres.shape}, label shape: {label.shape}")
    non_zero_idx =  torch.where(label.sum((1,2))>0)
    image_CTres=image_CTres[non_zero_idx]
    image_SUV=image_SUV[non_zero_idx]
    label=label[non_zero_idx]
    image_CTres+=label
    image_SUV+=label

    fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=200)
    ax1.imshow(montage(image_CTres), cmap='bone')
    plt.savefig('image_CTres.png')
    fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=200)
    ax1.imshow(montage(image_SUV), cmap='bone')
    plt.savefig('image_SUV.png')
    fig, ax1 = plt.subplots(1, 1, figsize=(40, 40), dpi=200)
    ax1.imshow(montage(label), cmap='bone')
    plt.savefig('label.png')
    return   train_loader,val_org_loader,post_transforms

if __name__ == '__main__':
    config = SimpleNamespace(patchshape=(96,96,96))
    train_loader,val_org_loader,post_transforms= get_distributed_data_loaders(config)
    for i in train_loader:
        print(i.keys())

