import glob,os
import SimpleITK as sitk
import numpy as np
from os.path import exists
import os
def read_CTres_and_SUV_image(patientPath):
    CTres_Path = os.path.join(patientPath, "CTres.nii.gz")
    imgCTres = sitk.ReadImage(CTres_Path)
    imgCTres = sitk.GetArrayFromImage(imgCTres)

    SUV_Path = os.path.join(patientPath, "SUV.nii.gz")
    imgSUV = sitk.ReadImage(SUV_Path)
    imgSUV = sitk.GetArrayFromImage(imgSUV)
    
    seg_Path = os.path.join(patientPath, "SEG.nii.gz")
    imgSEG = sitk.ReadImage(seg_Path)
    imgSEG = sitk.GetArrayFromImage(imgSEG)
    # return imgCTres,imgSUV,imgSEG
    return  np.concatenate((imgCTres,imgSUV),1) ,    imgSEG


# data_dir="data/FDG-PET-CT-Lesions/PETCT_04a4e1c874/11-18-2001-NA-PET-CT Ganzkoerper  primaer mit KM-96019"
data_dir = "data/FDG-PET-CT-Lesions/"
all_patients = os.listdir(data_dir)
all_patients =list(map(lambda x: os.path.join(data_dir,x),all_patients))
all_patients_path =[]
wealthy_p_count = 0

for i,patient in enumerate(all_patients):
    # print(i,patient)
    sub_patients = os.listdir(patient)
    for sub_patient in sub_patients:
        patient_dir = os.path.join(patient,sub_patient)
        print(patient_dir)
        image,label = read_CTres_and_SUV_image(patient_dir)
        train_image_path = os.path.join(patient_dir,"training_data_image.npy")
        # train_label_path = os.path.join(patient_dir,"training_data_label.npy")
        train_marker_path = os.path.join(patient_dir,"training_data_marker.npy")
        if (label.sum()==0):
            wealthy_p_count +=1
            np.save(train_image_path,image)
            # np.save(train_label_path,label)
            np.save(train_marker_path,np.array([0]))

            print(str(i)+"patient {} is  healthy shape is {}".format(patient,image.shape),flush=True)

        else:
            np.save(train_image_path,image)
            # np.save(train_label_path,label)
            np.save(train_marker_path,np.array([1]))

            print(str(i)+"patient {} is not healthy with labels {}  shape is {} ".format(patient,np.unique(label),image.shape),flush=True)
print(wealthy_p_count/len(all_patients))


    
