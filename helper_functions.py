import os
import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib
import ast

def save_image(frame, file_name, append = None, save_dir = '/kaggle/working/extracted_images'):
  base_dir = os.path.join(save_dir, str(file_name))
  os.makedirs(base_dir, exist_ok = True)
  if append is not None:
    file_name = f"{file_name}_{append}"
  save_path = os.path.join(base_dir, f'{file_name}.jpg')
  cv2.imwrite(save_path, frame)

def save_json(json_file, file_name, save_dir = '/kaggle/working/extracted_images'):
  save_dir = os.path.join(save_dir, str(file_name))
  os.makedirs(save_dir, exist_ok = True)
  file_dir = os.path.join(save_dir, 'yolo_metadata.json')
  with open(file_dir, 'w') as f:
    yolo_metadata = json.dumps(json_file)
    f.write(yolo_metadata)

def open_video(file_path):
  cap = cv2.VideoCapture(file_path)
  if not cap.isOpened():
    print("Error: Could not open video.")
  else:
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  return cap, frame_count

def extract_frame(cap_obj, frame_number=1, save_img=False, save_path=None):
    cap_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap_obj.read()
    
    if not ret:
        print(f"No fue posible extraer el frame {frame_number}")
        return None

    if save_img and save_path is not None:
        save_image(frame, save_path)

    return frame

def crop_and_save_image_train(array, detection_metadata, file_name, save_dir='./cropped_images', full_image = False, append = None):
    height, width = array.shape[:2]
    for bbx in detection_metadata:
        x_center, y_center, w, h = bbx[1:]
        x_min = x_center - w/2
        y_min = y_center - h/2
        x1 = int(x_min * width)
        y1 = int(y_min * height)
        x2 = int((x_min + w) * width)
        y2 = int((y_min + h) * height)

        crop_img = array[y1:y2, x1:x2]
        if full_image:
          to_save = array
        else:
          to_save = crop_img
        save_image(to_save, file_name, append=append, save_dir=save_dir)
        
def plot_labeled_image(array, detection_metadata, labeled_bbx):
    height, width = array.shape[:2]
    fig, ax = plt.subplots()
    ax.imshow(array)
    
    for obj in detection_metadata.values():
        x, y, w, h = obj["bbox"]
        x1, y1 = int(x * width), int(y * height)
        rect = patches.Rectangle(
            (x1, y1), int(w * width), int(h * height),
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
    if type(labeled_bbx[0]) == list:
    
        for obj in labeled_bbx:
          x_center, y_center, w, h = obj[1:]
          x_min = x_center - w/2
          y_min = y_center - h/2
          x1, y1 = int(x_min * width), int(y_min * height)
          rect2 = patches.Rectangle(
                (x1, y1), int(w * width), int(h * height),
                linewidth=2, edgecolor='green', facecolor='none'
          )
          ax.add_patch(rect2)
    else: 
        x_center, y_center, w, h = labeled_bbx[1:]
        x_min = x_center - w/2
        y_min = y_center - h/2
        x1, y1 = int(x_min * width), int(y_min * height)
        rect2 = patches.Rectangle(
                (x1, y1), int(w * width), int(h * height),
                linewidth=2, edgecolor='green', facecolor='none'
          )
        ax.add_patch(rect2)
    plt.axis('off')
    plt.show()
    
def plot_image(array, detection_metadata):
    height, width = array.shape[:2]
    fig, ax = plt.subplots()
    ax.imshow(array)
    
    for obj in detection_metadata.values():
        x, y, w, h = obj["bbox"]
        x1, y1 = int(x * width), int(y * height)
        rect = patches.Rectangle(
            (x1, y1), int(w * width), int(h * height),
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
    plt.axis('off')
    plt.show()
    
def bbx_from_txt(labeled_bbx):
  if labeled_bbx.find('\n') > -1:
    output_list = [list(map(float, line.split())) for line in labeled_bbx.strip().split('\n')]
  else: 
    output_list = ast.literal_eval('[' + labeled_bbx.replace(' ', ',') + ']')
  return output_list