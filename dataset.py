
"""
Created on Mon Feb  5 19:59:59 2024

@author: mary
"""
'''
1) get all sequences and their segmentation
2) extract tumor part
3) extract surrounding part

'''

# Paths to the image and segmentation directories

import numpy as np
from monai.transforms import (Compose, LoadImaged, AddChanneld, CropForegroundd,
                              ToTensord, Spacingd, ScaleIntensityD, 
                              Resized, RandFlipD, RandRotateD, RandZoomD)

from monai.data import Dataset, CacheDataset
from monai.transforms import MapTransform
from scipy.ndimage import binary_dilation

##############################################################################
class ExtractTumorMapTransform(MapTransform):
    """
    This transform applies the tumor extraction logic to each specified image modality 
    based on the segmentation mask, setting non-tumor parts to 0 for each modality.
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
    
    def __call__(self, data):
        # Generate the tumor mask based on the segmentation labels
        seg = data['seg']
        tumor_mask = (seg == 1) | (seg == 2) | (seg == 4)
        
        for key in self.keys:
            if key != 'seg':  # Skip segmentation mask
                # Apply the tumor mask to each modality, setting non-tumor regions to zero
                data[key] = data[key] * tumor_mask
        
        return data
#############################################################################
class ExtractTumorAndSurroundingMapTransform(MapTransform):
    """
    This transform dilates the tumor segmentation and applies it to each specified image modality,
    setting non-tumor parts and non-surrounding parts to 0 for each modality.
    """
    def __init__(self, keys, dilation_iters=10, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        
        self.dilation_iters = dilation_iters
    
    def __call__(self, data):
        seg = data['seg']
        tumor_mask = (seg == 1) | (seg == 2) | (seg == 4)
        
        # Dilate the tumor mask to include surrounding area
        dilated_tumor_mask = binary_dilation(tumor_mask, iterations=self.dilation_iters)
        
        for key in self.keys:
            if key !='seg':  
                data[key] = data[key] * dilated_tumor_mask
        return data
#############################################################################
class CustomTransformDataset(Dataset):
    def __init__(self, data_dicts, transform1, transform2, transform3):
        self.data_dicts = data_dicts
        self.transform1 = transform1  
        self.transform2 = transform2  
        self.transform3 = transform3  

    def __len__(self):
        return len(self.data_dicts)

    def __getitem__(self, idx):
        data_item = {key: self.data_dicts[idx][key] for key in ['t1', 't1ce', 't2', 'flair', 'seg']}
        
        
        transformed_data_tum = self.transform1(data_item)  
        transformed_data_surr1 = self.transform2(data_item)  
        transformed_data_surr2 = self.transform3(data_item)  
    
        
        return {'tum': transformed_data_tum, 'surr1': transformed_data_surr1, 'surr2': transformed_data_surr2}
##############################################################################
class CustomTransformCacheDataset(CacheDataset):
    def __init__(self, data_dicts, transform1, transform2, transform3, cache_rate=1.0, num_workers=2):
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3
        
        # Create a combined transform for MONAI CacheDataset
        self.transforms = Compose([self._transform_data])
        
        super().__init__(data=data_dicts, transform=self.transforms, cache_rate=cache_rate, num_workers=num_workers)

    def _transform_data(self, data_item):
        # Assuming data_item is a dictionary with keys for each modality and possibly 'seg'
        
        # Apply each transform
        transformed_data_tum = self.transform1(data_item)
        transformed_data_surr1 = self.transform2(data_item)
        transformed_data_surr2 = self.transform3(data_item)
        
        # Return the transformed data
        return {'tum': transformed_data_tum, 'surr1': transformed_data_surr1, 'surr2': transformed_data_surr2}    

##############################################################################
def make_ds(train_dict, val_dict, test_dict):
    
    keys = ["t1", "t2", "t1ce", "flair", "seg"]
    
    tumor_extract_transforms_tr = Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        ExtractTumorMapTransform(keys=["t1", "t2", "t1ce", "flair", "seg"]),
        CropForegroundd(keys=keys, source_key="seg", select_fn=lambda x: x > 0, margin=0),
        ScaleIntensityD(keys=keys),
        RandRotateD(keys=keys, range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandZoomD(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.5),
        Resized(keys=keys, spatial_size=[64, 64, 64]),
        ToTensord(keys=keys),
    ])
    
    tumor_extract_transforms_ts = Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        ExtractTumorMapTransform(keys=["t1", "t2", "t1ce", "flair", "seg"]),
        CropForegroundd(keys=keys, source_key="seg", select_fn=lambda x: x > 0, margin=0),
        ScaleIntensityD(keys=keys),
        Resized(keys=keys, spatial_size=[64, 64, 64]),
        ToTensord(keys=keys),
    ])
    
    surrounding_extract_transforms1_tr = Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=(1, 1, 1)),
        ExtractTumorAndSurroundingMapTransform(keys=["t1", "t2", "t1ce", "flair","seg"], dilation_iters=10),
        # Optionally include the CropForegroundd transform if you still want to crop around the foreground
        CropForegroundd(keys=keys, source_key="seg", select_fn=lambda x: x > 0, margin=10),
        ScaleIntensityD(keys=keys),
        RandRotateD(keys=keys, range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandZoomD(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.5),
        Resized(keys=keys, spatial_size=[64, 64, 64]),
        ToTensord(keys=keys),
    ])
    
    surrounding_extract_transforms1_ts = Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=(1, 1, 1)),
        ExtractTumorAndSurroundingMapTransform(keys=["t1", "t2", "t1ce", "flair","seg"], dilation_iters=10),
        # Optionally include the CropForegroundd transform if you still want to crop around the foreground
        CropForegroundd(keys=keys, source_key="seg", select_fn=lambda x: x > 0, margin=10),
        ScaleIntensityD(keys=keys),
        Resized(keys=keys, spatial_size=[64, 64, 64]),
        ToTensord(keys=keys),
    ])
    
    surrounding_extract_transforms2_tr = Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=(1, 1, 1)),
        ExtractTumorAndSurroundingMapTransform(keys=["t1", "t2", "t1ce", "flair","seg"], dilation_iters=5),
        # Optionally include the CropForegroundd transform if you still want to crop around the foreground
        CropForegroundd(keys=keys, source_key="seg", select_fn=lambda x: x > 0, margin=5),
        ScaleIntensityD(keys=keys),
        RandRotateD(keys=keys, range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandZoomD(keys=keys, min_zoom=0.9, max_zoom=1.1, prob=0.5),
        Resized(keys=keys, spatial_size=[64, 64, 64]),
        ToTensord(keys=keys),
    ])
    
    surrounding_extract_transforms2_ts = Compose([
        LoadImaged(keys=keys),
        AddChanneld(keys=keys),
        Spacingd(keys=keys, pixdim=(1, 1, 1)),
        ExtractTumorAndSurroundingMapTransform(keys=["t1", "t2", "t1ce", "flair","seg"], dilation_iters=5),
        # Optionally include the CropForegroundd transform if you still want to crop around the foreground
        CropForegroundd(keys=keys, source_key="seg", select_fn=lambda x: x > 0, margin=10),
        ScaleIntensityD(keys=keys),
        Resized(keys=keys, spatial_size=[64, 64, 64]),
        ToTensord(keys=keys),
    ])
    
    
    dataset_tr = CustomTransformCacheDataset(data_dicts=train_dict, 
                                     transform1=tumor_extract_transforms_tr,
                                     transform2=surrounding_extract_transforms1_tr,
                                     transform3=surrounding_extract_transforms2_tr)
    
    dataset_val = CustomTransformCacheDataset(data_dicts=val_dict, 
                                     transform1=tumor_extract_transforms_tr,
                                     transform2=surrounding_extract_transforms1_tr,
                                     transform3=surrounding_extract_transforms2_tr)
    
    dataset_ts = CustomTransformCacheDataset(data_dicts=test_dict, 
                                     transform1=tumor_extract_transforms_ts,
                                     transform2=surrounding_extract_transforms1_ts,
                                     transform3=surrounding_extract_transforms2_ts)
    
    return dataset_tr, dataset_val, dataset_ts

#############################################################################
