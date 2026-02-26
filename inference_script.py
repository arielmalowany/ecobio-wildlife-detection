# Import packages

import sys
sys.path.append('./yolov5')
import os
import torch
import numpy as np
import cv2
import pandas as pd
from speciesnet import DEFAULT_MODEL
from speciesnet import draw_bboxes
from speciesnet import load_rgb_image
from speciesnet import SpeciesNet
from speciesnet import SpeciesNetClassifier
from speciesnet import SUPPORTED_MODELS
from megadetector.visualization import visualization_utils as vis_utils
from megadetector.detection import run_detector
from helper_functions import *
from inference_functions import *
import torchvision.transforms.functional as F
from PIL import Image
from yolo_detector import yolo_inference

# Load models

custom_megadetector_model = run_detector.load_detector('./models/md_v5a.0.0.pt')
species_net = SpeciesNet('./models')
species_net_classifier_model = torch.load('./models/always_crop_99710272_22x8_v12_epoch_00148.pt', weights_only=False)
species_net_classifier_model.eval()
with open('./models/always_crop_99710272_22x8_v12_epoch_00148.labels.txt', mode="r", encoding="utf-8") as fp:
  labels = {idx: line.strip() for idx, line in enumerate(fp.readlines())}
  
with open('./models/species_dict.json', 'r') as species_dict:
  species_dict = json.load(species_dict)
  
# Iterate over a folder

predicciones = pd.DataFrame([], columns = ['Archivo', 'Prediccion', 'Confianza'])
videos_to_predict = [v for v in os.listdir('./videos') if any(ext in v for ext in ['.AVI', '.MP4'])]
for video in videos_to_predict:
    yolo = yolo_inference(
        detector_model = custom_megadetector_model,
        classifier_model = species_net,
        file_name = video,
        steps = 30,
        find_n_frames = 10
     )
    yolo_metadata = yolo.detect_and_predict_image()
    video_predictions = yolo_metadata.get('video_predictions')
    if len(video_predictions) == 0:
      row = pd.DataFrame([[video, 'no_object', 1]], columns = ['Archivo', 'Prediccion', 'Confianza'])
      predicciones = pd.concat((predicciones, row))
    else:
      for pred_class in video_predictions.keys():
        confidence = video_predictions.get(pred_class)
        row = pd.DataFrame([[video, pred_class, confidence]], columns = ['Archivo', 'Prediccion', 'Confianza'])
        predicciones = pd.concat((predicciones, row))
      
predicciones.to_excel('./predicciones/predicciones.xlsx')