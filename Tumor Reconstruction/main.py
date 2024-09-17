#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 19:49:41 2024

@author: mary
"""
import os
import glob
from sklearn.model_selection import train_test_split
from dataset import make_ds
from monai.data import DataLoader 
import matplotlib.pyplot as plt
import torch
from train_reconstruction import swinunetplussig, AEplussigmoid, train
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
image_dir = '/media/sdb/Maryam/brats21'

# Define patterns for matching file types
patterns = {
    't1': '*t1.nii.gz',
    't2': '*t2.nii.gz',
    'flair': '*flair.nii.gz',
    't1ce': '*t1ce.nii.gz',
    'seg': '*seg.nii.gz',
}

# Assuming the patient folders are directly under the image_dir
brats_folders = os.listdir(image_dir)

# Create a data dictionary
data_dicts = []
for patient in brats_folders:
    patient_folder = os.path.join(image_dir, patient)
    patient_dict = {}
    for modality, pattern in patterns.items():
        # Use glob to find files matching the pattern for each modality
        files = glob.glob(os.path.join(patient_folder, pattern))
        if files:
            # Assuming the first match is the correct one
            patient_dict[modality] = files[0]
        else:
            # Handle case where no matching file is found
            print(f"No matching file for {modality} in {patient_folder}")
    if len(patient_dict) == len(patterns):  # Ensure all modalities are found
        data_dicts.append(patient_dict)
    else:
        print(f"Missing modalities for patient {patient}, skipping...")
##############################################################################
#60 % for training, 10 % for validation and 30 % for test
train_dicts, temp_dicts = train_test_split(data_dicts, test_size=0.4, random_state=121274)

val_dicts, test_dicts = train_test_split(temp_dicts, test_size=0.75, random_state=121274)  

dataset_tr, dataset_val, dataset_ts = make_ds(train_dicts, val_dicts, test_dicts)

# dataloader_tr = DataLoader(dataset_tr, batch_size=1, shuffle=True)
# dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)
# dataloader_ts = DataLoader(dataset_ts, batch_size=1, shuffle=True)

dataloader_tr = DataLoader(dataset_tr, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
dataloader_ts = DataLoader(dataset_ts, batch_size=1, shuffle= False, num_workers=2, pin_memory=True)
##############################################################################
#visualizing data 

batch_data = next(iter(dataloader_ts))

# Define a function to plot images
def plot_images(batch, modality_keys, slice_idx=30):
    """
    Plots a single slice from each modality for the first item in the batch.
    
    Parameters:
    - batch: the batch of data from the DataLoader.
    - modality_keys: list of keys for each modality to plot.
    - slice_idx: the index of the slice to plot.
    """
    fig, axs = plt.subplots(1, len(modality_keys), figsize=(15, 5))
    for i, key in enumerate(modality_keys):
        # Assuming the first dimension is the batch size and images are 4D (B, C, H, W, D)
        image = batch['surr2'][key][0, 0, :, :, slice_idx].detach().cpu().numpy()  # Get the first item, first channel
        axs[i].imshow(image, cmap='gray')
        axs[i].set_title(key)
        axs[i].axis('off')
    plt.show()

# Define the keys for the modalities you want to plot
modality_keys = ["t1", "t2", "t1ce", "flair", "seg"]

# Plot images for a selected slice
plot_images(batch_data, modality_keys, slice_idx=50)
##############################################################################

device = torch.device("cuda:0" )#if torch.cuda.is_available() else "cpu")
# network = swinunetplussig(device)
network = swinunetplussig(device)

epoch_loss_values, epoch_loss_values_val, model = train(device, dataloader_tr, dataloader_val, network, epochs=100, learning_rate=1e-4)

data = {
    'Training Loss': epoch_loss_values,
    'Validation Loss': epoch_loss_values_val
}

torch.save(model.state_dict(), os.path.join('/media/sdb/Maryam/surrounding_prognosis/saved_models','Reconstruction_epch_last.pth'))
df_losses = pd.DataFrame(data)
df_losses.to_csv('losses.csv', index=False)