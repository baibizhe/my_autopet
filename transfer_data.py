import glob,os
import SimpleITK as sitk
import numpy as np
from os.path import exists
import os
def read_CTres_and_SUV_image(patientPath):
    CTres_Path = os.path.join(patientPath, "CTres.nii.gz")
    imgCTres = sitk.ReadImage(CTres_Path)
    imgCTres = np.expand_dims(sitk.GetArrayFromImage(imgCTres),axis=0)

    SUV_Path = os.path.join(patientPath, "SUV.nii.gz")
    imgSUV = sitk.ReadImage(SUV_Path)
    imgSUV = np.expand_dims(sitk.GetArrayFromImage(imgSUV),axis=0)
    
    seg_Path = os.path.join(patientPath, "SEG.nii.gz")
    imgSEG = sitk.ReadImage(seg_Path)
    imgSEG = np.expand_dims(sitk.GetArrayFromImage(imgSEG),axis=0)
    # return imgCTres,imgSUV,imgSEG
    return  np.concatenate((imgCTres,imgSUV),0) ,    imgSEG


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
        SUV_path = os.path.join(patient_dir,"SUV.nii.gz")
        CTres_path = os.path.join(patient_dir,"CTres.nii.gz")
        if os.path.exists(SUV_path):
            os.remove(SUV_path)
        if os.path.exists(CTres_path):
            os.remove(CTres_path)

        if (label.sum()==0):
            wealthy_p_count +=1
            np.save(train_image_path,image)
            # np.save(train_label_path,label)
            np.save(train_marker_path,np.array([0]))

            print(str(i)+"patient {} is  healthy , image shape is {} label shape is {}".format(patient,image.shape , label.shape),flush=True)

        else:
            np.save(train_image_path,image)
            # np.save(train_label_path,label)
            np.save(train_marker_path,np.array([1]))

            print(str(i)+"patient {} is unhealthy , image shape is {} label shape is {}".format(patient,image.shape , label.shape),flush=True)
print(wealthy_p_count/len(all_patients))


    
