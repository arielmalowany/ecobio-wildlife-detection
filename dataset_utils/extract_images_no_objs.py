import os 
import cv2
import numpy as np

def open_video(file_name):
    video_path = os.path.join('/Users/arielmalowany/Desktop/Learning/Cupybara/dataset/manually curated/no - object', file_name)
    cap = cv2.VideoCapture(video_path)
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
        cv2.imwrite(save_path, frame)

    return frame

no_object_path = '/Users/arielmalowany/Desktop/Learning/Cupybara/dataset/manually curated/no - object'
no_object_videos = os.listdir(no_object_path)
no_object_videos.remove('.DS_Store')
save_folder = '/Users/arielmalowany/Desktop/Learning/Cupybara/training_dataset/no object'
for file_name in no_object_videos:
  cap_obj, frame_count = open_video(file_name)
  frames_to_extract = np.random.randint(frame_count - 1, size = 2)
  for f in frames_to_extract:
    save_path = os.path.join(save_folder, file_name.replace('.mp4', '') + '_' + str(f) + '.jpg')
    extract_frame(cap_obj, frame_number=f, save_img=True, save_path=save_path)