
import os
import torch
import numpy as np
import cv2
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
  
# Load Detector and Classifier

def run_inference(
             file_name, 
             file_extension,
             steps,
             find_n_frames,
             detector_model, 
             classifier_model,
             species_dict
    ):

    yolo = yolo_inference(
            detector_model=detector_model,
            classifier_model = classifier_model,
            file_name = file_name,
            file_extension = file_extension,
            steps = steps,
            find_n_frames = find_n_frames
            )

    yolo_metadata = yolo.detect_and_predict_image()
    speciesnet_preds = species_net_to_cupybara(yolo_metadata, species_dict)
    prediction = final_predict(yolo_metadata, speciesnet_preds)
    return prediction