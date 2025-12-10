import shutil
import os
import pandas as pd

# Base directory where the .mp4 files are
base_dir = "/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/images/train"

# Subfolder to copy files into (create if it doesn't exist)
destination = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/augmented images'

# Import pandas df

data_dic = pd.read_excel('/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/Data_Dictionary.xlsx', sheet_name = 'Training Set')

filenames = list(data_dic.loc["Training File"])

# Copy each file
for imgs in filenames:
  if imgs.endswith(".jpg"):
    src = os.path.join(base_dir, imgs)
    dst = os.path.join(destination, imgs)
            
  if not os.path.exists(dst):
    shutil.copy(src, dst)
    print(f"Copied: {imgs}")
  else:
    print(f"Skipped (already exists): {imgs}")