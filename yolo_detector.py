from helper_functions import *
from inference_functions import *
import numpy as np
import os

class yolo_inference:
    def __init__(
        self, 
        detector_model = None,
        classifier_model = None,
        steps = 30,
        find_n_frames = 3,
        threshold = 0.5,
        not_indexes = None,
        videos_path = './videos',
        save_dir_path = './cropped_images',
        file_name = None,
        file_extension = 'avi'
        ):
        
        self.detector_model = detector_model
        self.classifier_model = classifier_model
        self.steps = steps
        self.find_n_frames = find_n_frames
        self.threshold = threshold
        self.not_indexes = not_indexes
        self.save_path = save_dir_path
        self.file_name = file_name
    
        self.cap_obj, self.frame_count = open_video(os.path.join(videos_path, file_name))
        
        if self.not_indexes is None:
            self.indexes = np.linspace(0, self.frame_count - 1, self.steps, dtype=int)
            self.iterated_frames = self.indexes.tolist()
        else:
            frame_seq = list(range(self.frame_count - 1))
            frame_seq = list(set(frame_seq) - set(not_indexes))
            frame_idx = list(np.linspace(0, len(frame_seq) - 1, steps, dtype=int))
            self.indexes = [frame_seq[i] for i in frame_idx]
            self.iterated_frames = not_indexes + self.indexes
    
        self.save_file_name = file_name.replace('.' + file_extension, '')
        
    def detect_and_predict_image(self):
      indexes = self.indexes
      find_n_frames = self.find_n_frames
      cap_obj = self.cap_obj
      frame_count = self.frame_count
      threshold = self.threshold
      iterated_frames = self.iterated_frames
      save_file_name = self.save_file_name
      save_dir_path = self.save_path
      
      image_yolo_metadata = {}
      video_predictions = {}
      
      frames_with_objects = 0
      i = 0
      category = []
      while i < len(indexes) and frames_with_objects < find_n_frames:
        frame_num = indexes[i]
        frame = extract_frame(cap_obj, frame_num)
        if frame is None:
            continue
        
        detection_metadata = detect_image_objects(frame, threshold = threshold, model = self.detector_model)
        if detection_metadata:
           frames_with_objects += 1
           crop_and_save_image(frame, detection_metadata, self.save_file_name, frame_num, classifier_model = self.classifier_model, return_dict=True, save_dir = save_dir_path)
           image_yolo_metadata[str(frame_num)] = detection_metadata
           for obj in list(detection_metadata.keys()):
               obj_metadata = detection_metadata.get(obj)
               video_pred = obj_metadata["pred_class"]
               category.append(int(obj_metadata["category"]))
               found_classes = video_predictions.keys()
               for c, s in video_pred:
                 if c not in found_classes:
                     video_predictions[c] = s
                 max_score = video_predictions[c]
                 if s > max_score:
                  video_predictions[c] = s
                  
        i +=1
        if frames_with_objects > 0:
          image_yolo_metadata["category"] = np.mean(category)
        image_yolo_metadata["frames_with_objects"] = frames_with_objects
        image_yolo_metadata["iterated_frames"] = iterated_frames
        image_yolo_metadata["video_predictions"] = dict(sorted(video_predictions.items(), key=lambda item: item[1], reverse = True))
        save_json(image_yolo_metadata, save_file_name, save_dir_path)

        return image_yolo_metadata