import os

import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import numpy as np

def read_CTres_and_SUV_image(patientPath):
    images= np.load(os.path.join(patientPath,"images.npy"))
    return  images


def read_label(patientPath):
    label= np.load(os.path.join(patientPath,"segmentation.npy"))
    return  label


# def read_CTres_and_SUV_image(patientPath):
#     CTres_Path = os.path.join(patientPath, "CTres.nii.gz")
#     imgCTres = sitk.ReadImage(CTres_Path)
#     imgCTres = np.expand_dims(sitk.GetArrayFromImage(imgCTres),0)
#
#     SUV_Path = os.path.join(patientPath, "CTres.nii.gz")
#     imgSUV = sitk.ReadImage(SUV_Path)
#     imgSUV = np.expand_dims(sitk.GetArrayFromImage(imgSUV),0)
#     return  np.concatenate((imgCTres,imgSUV),0)
#
#
# def read_label(patientPath):
#     SEG_Path = os.path.join(patientPath, "SEG.nii.gz")
#     imgSEG = sitk.ReadImage(SEG_Path)
#     imgSEG = sitk.GetArrayFromImage(imgSEG)
#     return  np.expand_dims(imgSEG,0)


class CustomTrainImageDataset(Dataset):
    def __init__(self, allImagePath, augmentation=None):
        self.allPatientsPath = allImagePath
        self.augmentation = augmentation

    def __len__(self):
        return len(self.allPatientsPath)

    def __getitem__(self, idx):
        image = read_CTres_and_SUV_image(self.allPatientsPath[idx])
        label = read_label(self.allPatientsPath[idx])

        if self.augmentation:
            data = {"image":image,"label":label}
            dataAugmented = self.augmentation(data)
            image = dataAugmented["image"]
            label = dataAugmented["label"]
        label = torch.nn.functional.one_hot(torch.tensor(label).to(torch.int64),num_classes=2)
        label = label.transpose(1,-1).squeeze(-1)
        return torch.tensor(image),label
