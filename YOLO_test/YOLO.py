from ultralytics import YOLO
from typing import List, Dict, Tuple
import numpy as np
threshold = .5
def filter_results(results) -> List[List[str]]:
    """
    Filter the results of the YOLO model to separate detected objects into left, forward, and right categories.
    :param results: Results object from YOLO model.
    """
    left, forward, right = [], [], []
    for result in results:
        if len(result.boxes) == 0:
            continue
        objects = result.to_df()['name'].to_list()
        boxes = result.boxes  # Boxes object for bounding box outputs
        coords = boxes.xywhn  # normalized xywh (x_center, y_center, width, height)
        probs = boxes.conf  # Bounding box coordinates xyxy (Top-left x, Top-left y, Bottom-right x, Bottom-right y)
        for prob, box, i in zip(probs, coords, range(len(objects))):
            if prob < 0.5:
                continue

            x_center, y_center, width, height = box
            if x_center < 0.33:
                left.append(objects[i])
            elif x_center > 0.66:
                right.append(objects[i])
            else:
                forward.append(objects[i])
        
    return left, forward, right

def count_objects(objects: List[str]) -> Dict[str, int]:
    """
    Count the frequency of each detected object.
    :param objects: List of detected objects.
    :return: Dictionary with object names as keys and their counts as values.
    """
    object_counts = {}
    for obj in objects:
        if obj in object_counts:
            object_counts[obj] += 1
        else:
            object_counts[obj] = 1
    return object_counts


def yolo_object_detection_v11(img_rgb: np.ndarray) -> Tuple[Dict[str, int]]:
    """
    Perform object detection using YOLO11.
    :param img_rgb: RGB image as a numpy array.
    :return: Tupe of left, forward, and right dictionary of detected objects.
    """
    # Load a pre-trained YOLO11 model (e.g., YOLO11n)
    model = YOLO('yolo11m.pt')

    # Perform inference on an image
    results = model(img_rgb)
    left_objects, forward_objects, right_objects = filter_results(results)

    return count_objects(left_objects), count_objects(forward_objects), count_objects(right_objects)

def object_description_generator(detected_objects:  Dict[str, int]) -> List[str]:
    """
    Generate object descriptions based on detected objects.
    :param detected_objects: Dictionary of detected objects.
    :return: List of detected objects with descriptions.
    """
    # Generate descriptions for each detected object
    object_descriptions = []
    for obj, freq in detected_objects.items():
        if freq > 1:
            description = f"There are {freq} {obj}s"
        else:
            description = f"There is a {obj}"
        object_descriptions.append(description)

    return object_descriptions if len(object_descriptions) > 0 else ["No objects detected"]

if __name__ == "__main__":
    # Example usage
    image_path = "chairs.jpg"  # Replace with your image path
    detected_objects = yolo_object_detection_v11(image_path)
    print("Detected objects:", detected_objects)


