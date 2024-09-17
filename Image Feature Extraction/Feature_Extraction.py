from tqdm import tqdm
import numpy as np
import torch
from train_reconstruction import swinunetplussig, lossf
from dataset import make_ds_test
import os
from monai.data import DataLoader 
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = swinunetplussig(device)
model_path = '/media/sdb/Maryam/surrounding_prognosis/saved_models/Reconstruction_epch_last.pth'
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Initialize empty lists to store the extracted features for each category
features_tum = []
features_surr1 = []
features_surr2 = []


root_dir = '/media/sdb/Maryam/Brats_images/Brats_surv'
image_dir = os.path.join(root_dir, 'images')
label_dir = os.path.join(root_dir, 'labels')
data_dict1 = []
brats20_ids = []


for filename in os.listdir(image_dir):
    # Extract the patient ID from the filename
    patient_id = filename.split('.')[0]  
    patient_id = patient_id.replace('_t2', '')
    brats20_ids.append(patient_id)
    
    # Construct the file paths for the image and label
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, f"{patient_id}_seg.nii.gz") 
    
    # Create a dictionary for the patient
    patient_dict = {
        't2': image_path,
        'seg': label_path
    }
    
    # Append the patient dictionary to the data list
    data_dict1.append(patient_dict)

data_dict1 = sorted(data_dict1, key=lambda x: x['t2'])
brats20_ids = sorted(brats20_ids)
dataset_brats20 = make_ds_test(data_dict1)
dataloader_brats20 = DataLoader(dataset_brats20, batch_size=1, shuffle= False, num_workers=2, pin_memory=True)

# Iterate over the patients in the dataloader
for patient_data in tqdm(dataloader_brats20, desc="Extracting features"):
    # Extract features for each input tensor (tum, surr1, surr2)
    for key in ['tum', 'surr1', 'surr2']:
        input_tensor = patient_data[key]['t2'].to(device)
        
        with torch.no_grad():
            features = model.ex_features(input_tensor)
        
        # Convert the features to a numpy array and append to the corresponding list
        if key == 'tum':
            features_tum.append(features.cpu().numpy())
        elif key == 'surr1':
            features_surr1.append(features.cpu().numpy())
        elif key == 'surr2':
            features_surr2.append(features.cpu().numpy())

# Convert the lists of features to 2D numpy arrays
features_tum = np.concatenate(features_tum, axis=0)
features_surr1 = np.concatenate(features_surr1, axis=0)
features_surr2 = np.concatenate(features_surr2, axis=0)

features_tum_vector = torch.flatten(torch.from_numpy(features_tum), start_dim=1)
features_surr1_vector = torch.flatten(torch.from_numpy(features_surr1), start_dim=1)
features_surr2_vector = torch.flatten(torch.from_numpy(features_surr2), start_dim=1)

df_features_tum = pd.DataFrame(features_tum_vector.numpy())
df_features_surr1 = pd.DataFrame(features_surr1_vector.numpy())
df_features_surr2 = pd.DataFrame(features_surr2_vector.numpy())

# Save the DataFrames as CSV files
output_path_tum = '/media/sdb/Maryam/surrounding_prognosis/features_tum.csv'
output_path_surr1 = '/media/sdb/Maryam/surrounding_prognosis/features_surr1.csv'
output_path_surr2 = '/media/sdb/Maryam/surrounding_prognosis/features_surr2.csv'

df_features_tum.to_csv(output_path_tum, index=False)
df_features_surr1.to_csv(output_path_surr1, index=False)
df_features_surr2.to_csv(output_path_surr2, index=False)
