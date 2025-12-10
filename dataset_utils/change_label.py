"""Import YOLO / MEGADETECTOR Labels and perform operations if needed"""

import os

label_dir = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/labels/test'  # adjust if needed

for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        filepath = os.path.join(label_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts and len(parts) == 5:
                class_id = str(int(float(parts[0])))
                x_center = float(parts[1])
                y_center = float(parts[2])
                box_width = float(parts[3])
                box_height = float(parts[4])

                #x_center = xmin + box_width / 2
                #y_center = ymin + box_height / 2

                new_line = f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
                new_lines.append(new_line)

        with open(filepath, 'w') as f:
            f.write('\n'.join(new_lines) + '\n')
