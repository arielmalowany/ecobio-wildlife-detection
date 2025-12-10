import shutil
import os
import numpy as np

# Base directory where the images are
base_dir = "/Users/arielmalowany/Desktop/Learning/Cupybara/manually labeled/images/test"
files_to_move = os.listdir(base_dir)

# Subfolder to copy files into (create if it doesn't exist)
destination_train = '/Users/arielmalowany/Desktop/Learning/Cupybara/manually labeled/images/test'
destination_val = '/Users/arielmalowany/Desktop/Learning/Cupybara/manually labeled/images/val'
#os.mkdir(destination_train) 
#os.mkdir(destination_val) 

# Train-val percentages
train = 0.5
val = 1 - train
n = len(files_to_move)
n_train = int(np.floor(n * train))

all_indices = np.arange(n)
np.random.shuffle(all_indices)

train_idx = all_indices[:n_train]
val_idx = all_indices[n_train:]

#for idx in train_idx:
#  src_path = os.path.join(base_dir, files_to_move[idx])
#  dst_path = os.path.join(destination_train, files_to_move[idx])
#  shutil.copy(src_path, dst_path)
  
for idx in val_idx:
  src_path = os.path.join(base_dir, files_to_move[idx])
  dst_path = os.path.join(destination_val, files_to_move[idx])
  shutil.move(src_path, dst_path)
   