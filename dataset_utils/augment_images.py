"""Create augmented versions of images to increase the dataset's diversity"""

import cv2
import os
import albumentations as A
import glob

# Set paths
IMAGE_DIR = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/augmented images'
LABEL_DIR = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/augmented labels'
OUTPUT_IMAGE_DIR = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/augmented images'
OUTPUT_LABEL_DIR = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/augmented labels'

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# Number of augmentations per image
NUM_AUGS = 2

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-15, 15), p=0.7),
    A.Blur(blur_limit=3, p=0.3),
    A.HueSaturationValue(p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Function to load YOLO labels
def load_yolo_labels(txt_path):
    with open(txt_path, 'r') as f:
        labels = f.read().splitlines()
    bboxes = []
    class_labels = []
    for label in labels:
        cls, x, y, w, h = map(float, label.split())
        bboxes.append([x, y, w, h])
        class_labels.append(int(cls))
    return bboxes, class_labels

# Augment each image
image_paths = glob.glob(f"{IMAGE_DIR}/*.jpg")
for img_path in image_paths:
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    filename = os.path.basename(img_path).rsplit('.', 1)[0]
    label_path = os.path.join(LABEL_DIR, filename + '.txt')

    if not os.path.exists(label_path):
        continue

    bboxes, class_labels = load_yolo_labels(label_path)

    for i in range(NUM_AUGS):
        try:
            augmented = transform(image=img, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']

            aug_img_path = os.path.join(OUTPUT_IMAGE_DIR, f"{filename}_aug{i}.jpg")
            aug_label_path = os.path.join(OUTPUT_LABEL_DIR, f"{filename}_aug{i}.txt")

            cv2.imwrite(aug_img_path, aug_img)

            with open(aug_label_path, 'w') as f:
                for bbox, cls in zip(aug_bboxes, aug_class_labels):
                    x, y, w, h = bbox
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        except Exception as e:
            print(f"Skipping {filename} aug {i}: {e}")