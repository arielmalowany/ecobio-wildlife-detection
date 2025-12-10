from helper_functions import *
import numpy as np

def retrieve_prediction(img, frame, obj, classifier_model = None):
    try:
        predictions_dict = classifier_model.classify(
            filepaths=[f'./cropped_images/{img}/{img}_{frame}_{obj}.jpg'],
            country = ['ARG', 'BRA', 'PRY', 'URY']
        )

        preds = predictions_dict["predictions"][0]
        if "classifications" not in preds:
            return "unknown", 0.0

        scores = preds["classifications"]["scores"]
        classes = preds["classifications"]["classes"]

        return [(x, y) for x, y in zip(classes, scores)]

    except Exception as e:
        print(f"Prediction failed for {img}_{frame}_{obj}: {e}")
        return "error", 0.0

def crop_and_save_image(array, detection_metadata, file_name, frame_number, classifier_model = None,
                        return_dict=False, save_dir='./cropped_images', full_image = False):
    height, width = array.shape[:2]
    for idx_str, obj in detection_metadata.items():
        bbx = obj["bbox"]
        x1 = int(bbx[0] * width)
        y1 = int(bbx[1] * height)
        x2 = int((bbx[0] + bbx[2]) * width)
        y2 = int((bbx[1] + bbx[3]) * height)

        crop_img = array[y1:y2, x1:x2]
        append = f"{frame_number}_{idx_str}"
        if full_image:
          to_save = array
        else:
          to_save = crop_img
        save_image(to_save, file_name, append=append, save_dir=save_dir)

        predictions = retrieve_prediction(file_name, str(frame_number), idx_str, classifier_model)
        if return_dict:
            obj["pred_class"] = predictions

    if return_dict:
        return detection_metadata

def detect_image_objects(frame, threshold=0.2, model = None):
    result = model.generate_detections_one_image(frame)
    detections = result.get('detections', [])
    # List comprehension for filtered detections, use dict comprehension for concise creation
    filtered = {str(i): {"category": d["category"], "confidence": d["conf"], "bbox": d["bbox"]}
                for i, d in enumerate(detections) if d["conf"] > threshold}
    return filtered

def species_net_to_cupybara(yolo_metadata, species_dict):
    yolo_dict = yolo_metadata
    yolo_keys = yolo_dict.keys()
    video_predictions = yolo_dict["video_predictions"]
    species_list = species_dict.keys()
    mapped_predicted_species = {}
    for pred_class, score in video_predictions.items():
        for species in species_list:
            if species in pred_class:
                label = species_dict[species]
                if label not in mapped_predicted_species.keys() and score >= 0.05:
                    mapped_predicted_species[label] = score
                if score >= 0.05 and mapped_predicted_species[label] < score:
                    mapped_predicted_species[label] = score
    return mapped_predicted_species

def final_predict(yolo_metadata, speciesnet_preds):
    frames_with_objects = yolo_metadata.get('frames_with_objects', 0)
    md_category = yolo_metadata.get('category')
    prediction = 'no_object'

    if frames_with_objects == 0:
        return prediction  # early return for no objects

    predicted_classes = list(speciesnet_preds.keys())
    unk_score = speciesnet_preds.get('unknown_animal', 0)

    # Filter out 'unknown_animal'
    no_unk_preds = [cls for cls in predicted_classes if cls != 'unknown_animal']

    if not no_unk_preds:
        if md_category == 1:
            return 'unknown_animal'
        elif md_category == 2:
            return 'human'
        return prediction  # default: no_object (though this case may never occur)

    # Top prediction excluding 'unknown_animal'
    top_no_unk_class = no_unk_preds[0]
    top_no_unk_score = speciesnet_preds.get(top_no_unk_class, 0)

    # Heuristic thresholds
    all_no_unk_scores = [speciesnet_preds[cls] for cls in no_unk_preds]
    confounded_threshold = np.mean(all_no_unk_scores) + 2.5 * np.std(all_no_unk_scores)

    # Bird-related logic
    bird_like = {'dusky_legged_guan', 'bird'}
    confounded_birds = bird_like | {'squirrel'}
    birds_in_preds = [cls for cls in no_unk_preds if cls in bird_like]
    confounded_in_preds = [cls for cls in no_unk_preds if cls in confounded_birds]

    # Prediction logic
    if len(predicted_classes) == 1:
        prediction = predicted_classes[0]
    elif top_no_unk_score > unk_score and top_no_unk_score >= 0.25:
        prediction = top_no_unk_class
    elif len(top_no_unk_class) == 1 and top_no_unk_score >= 0.10:
        prediction = top_no_unk_class
    elif set(no_unk_preds) == bird_like:
        prediction = 'dusky_legged_guan'
    elif set(no_unk_preds) == confounded_birds:
        prediction = 'bird'
    elif unk_score > top_no_unk_score:
        if top_no_unk_score > 0.50 or top_no_unk_score > confounded_threshold:
            prediction = top_no_unk_class
        else:
            prediction = 'unknown_animal'
    else:
        prediction = top_no_unk_class  # fallback if none of the above rules hit

    # Normalize "unwanted" classes
    if prediction in {'domestic_cat', 'fox', 'squirrel'}:
        prediction = 'unknown_animal'

    return prediction