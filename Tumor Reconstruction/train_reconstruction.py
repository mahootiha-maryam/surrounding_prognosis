#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:59:02 2024
@author: mary
"""
from swin_unetr import  SwinUNETR
import torch
from tqdm import trange
import torch.nn.functional as F
from pathlib import Path
import os 
from pytorch_msssim import ssim


class addsigmoid(SwinUNETR):
    def forward(self, x):
        x = super().forward(x)
        return torch.sigmoid(x)
    def ex_features(self, x):
        return super().ex_features(x)
    
# def swinunetplussig(device):
#     # network = addsigmoid(img_size=(128,128,128), in_channels=1, out_channels=1).to(device)
#     network = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=1).to(device)
#     return network

def swinunetplussig(device):
    network = addsigmoid(img_size=(64,64,64), in_channels=1, out_channels=1,
                         depths=(1, 1, 1, 1),  feature_size=12,
                         num_heads=(3, 3, 3, 3)).to(device)
    return network

# class CustomAutoEncoder(AutoEncoder):
#     def forward(self, x):
#         x = super().forward(x)
#         return torch.sigmoid(x)
      
# def AEplussigmoid(device):
#     network = CustomAutoEncoder(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(64, 128, 256, 512, 1024),
#     strides=(2, 2, 2, 2, 2),).to(device)
#     return network

def lossf(img1,img2):
    '''
    Adding structural similarity index measure (SSIM) to mean squared error 
    (MSE) as a combined loss function is a common practice in image 
    reconstruction tasks, as it provides a balance between low-level pixel 
    difference (MSE) and perceived visual difference (SSIM).
    Because of scaleintensity transformation in monai all of image values
    are between 0 and 1 so mse is between 0 and 1 the best is 1. The best 
    of ssim is 1 the worst is -1.
    '''
    ssim_value = ssim(img1, img2, data_range=1.0, size_average=True)
    mse_value = F.mse_loss(img1, img2)
    loss = mse_value - ssim_value

    return loss
    
def train(device, train_loader, val_loader, model, epochs=10, learning_rate=1e-4):

    # Create loss fn and optimiser
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    epoch_loss_values = []
    epoch_loss_values_val = []

    t = trange(
        epochs,
        desc=" epoch 0, avg loss: inf", leave=True)
    
    val_interval = 1
    for epoch in t:
        model.train()
        epoch_loss = 0

        
        for batch_data in train_loader:
            optimizer.zero_grad()
            total_loss = 0  # Aggregate loss for each batch

            # Loop over each condition and sequence
            for condition in ['tum', 'surr1', 'surr2']:
                for sequence in ['t1', 't2', 't1ce', 'flair']:
                    # Process each sequence-condition pair through the model
                    inputs = batch_data[condition][sequence].to(device)
                    
                    outputs = model(inputs)

                    # Calculate loss for this sequence-condition pair
                    loss = lossf(outputs, inputs)
                    total_loss += loss  # Aggregate the loss

            # Perform backward pass and optimizer step after aggregating losses
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        epoch_loss /= len(train_loader)
        epoch_loss_values.append(epoch_loss)
        t.set_description(f"Epoch {epoch+1}, avg loss: {epoch_loss:.4f}")
        
        
        if epoch % val_interval == 0:
            model.eval()
            step_test = 0
            epoch_loss_val = 0
            with torch.no_grad():
                for val_data in val_loader:
                    step_test += 1
                    for condition in ['tum', 'surr1', 'surr2']:
                        for sequence in ['t1', 't2', 't1ce', 'flair']:
                            
                            inputs_val = val_data[condition][sequence].to(device)
                    
                            outputs_val = model(inputs_val)
                            loss_val = lossf(outputs_val, inputs_val)
                            epoch_loss_val += loss_val.item()
        epoch_loss_val /= len(val_loader)
        epoch_loss_values_val.append(epoch_loss_val)
        
        t.set_description(
            f"-- epoch {epoch + 1}"
            + f", average train loss: {epoch_loss:.4f}" + f", average test loss:{epoch_loss_val:.4f}")
        
        if epoch % 10 == 0:
            path_savedmodels = Path('/media/sdb/Maryam/surrounding_prognosis/saved_models')
            torch.save(model.state_dict(), os.path.join(path_savedmodels,f'Reconstruction_epch{epoch}.pth'))
            
    return epoch_loss_values, epoch_loss_values_val, model
            
