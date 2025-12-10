import os
import shutil

training_path = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/images/test'
destination_path = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/images/test_deduped'
files = os.listdir(training_path)

for file in files:
  new_file_name = "_".join(file.split("_")[:-1]) + ".jpg"
  original_file = os.path.join(training_path, file)
  new_file = os.path.join(destination_path, new_file_name)
  if not os.path.exists(new_file):
    shutil.move(original_file, new_file)
