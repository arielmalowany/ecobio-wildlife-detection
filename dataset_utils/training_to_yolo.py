import os 
import pandas as pd
import ast
os.chdir('/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/labels/all_labels')

training_set_pd = pd.read_excel('/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/curated_training_set.xlsx')
training_set_pd['bbox'] = training_set_pd['bbox'].apply(lambda x: ast.literal_eval(x.strip()) if isinstance(x, str) else x)
training_set_pd = training_set_pd.groupby(['file', 'frame'])
for (file_id, frame), group in training_set_pd:
  lines = []
  for _, row in group.iterrows():
    class_id = row['category']
    x_min, y_min, w, h = row['bbox']
    x = x_min + w / 2
    y = y_min + h / 2
    lines.append(f"{class_id} {x} {y} {w} {h}")
  filename = f"{file_id}_{frame}.txt"
  with open(filename, 'w') as f:
    f.write("\n".join(lines))