import cv2
import numpy as np
import os
import argparse

def padding(image, patch_size):
    height, width = image.shape[:2]
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

def is_deprecated(image_array, max_blank_ratio):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) / 255.0
    return np.mean(gray) > max_blank_ratio

def process_bci_dataset(input_path, output_path):
    patch_size = 256
    for split in ['train', 'validate', 'test']:
        split_in = os.path.join(input_path, split)
        split_out = os.path.join(output_path, split)
        
        if not os.path.exists(split_in): continue
        
        for category in ['HER2-0', 'HER2-1+', 'HER2-2+', 'HER2-3+']:
            cat_in = os.path.join(split_in, category)
            cat_out = os.path.join(split_out, category)
            os.makedirs(cat_out, exist_ok=True)
            
            if not os.path.exists(cat_in): continue
            
            for img_name in os.listdir(cat_in):
                if img_name.startswith('.'): continue
                img = cv2.imread(os.path.join(cat_in, img_name))
                if img is None: continue
                
                img = padding(img, patch_size)
                h, w = img.shape[:2]
                for y in range(0, h, patch_size):
                    for x in range(0, w, patch_size):
                        patch = img[y:y+patch_size, x:x+patch_size]
                        if not is_deprecated(patch, 0.8):
                            save_name = f"{os.path.splitext(img_name)[0]}_{x}_{y}.jpg"
                            cv2.imwrite(os.path.join(cat_out, save_name), patch)

if __name__ == "__main__":
    # Update these paths to your actual local folders
    RAW_DATA = './BCI_Raw' 
    PATCH_OUT = './BCI_Patches'
    process_bci_dataset(RAW_DATA, PATCH_OUT)
    print("Patch extraction complete.")
