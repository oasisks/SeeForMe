from ultralytics import YOLO
from typing import List
import numpy as np

def yolo_object_detection_v11(img_rgb: np.ndarray) -> List[str]:
    """
    Perform object detection using YOLO11.
    :param img_rgb: RGB image as a numpy array.
    :return: List of detected objects.
    """
    # Load a pre-trained YOLO11 model (e.g., YOLO11n)
    model = YOLO('yolo11m.pt')

    # Perform inference on an image
    results = model(img_rgb)
    return results[0].to_df()['name'].to_list()

def object_description_generator(img_rgb: np.ndarray) -> List[str]:
    """
    Generate object descriptions from an image using YOLO11.
    :param img_rgb: RGB image as a numpy array.
    :return: List of detected objects with descriptions.
    """
    # Get the detected objects
    detected_objects = yolo_object_detection_v11(img_rgb)

    # Find frequencies of detected objects
    object_frequencies = {}
    for obj in detected_objects:
        if obj in object_frequencies:
            object_frequencies[obj] += 1
        else:
            object_frequencies[obj] = 1

    # Generate descriptions for each detected object
    object_descriptions = []
    for obj, freq in object_frequencies.items():
        if freq > 1:
            description = f"There are {freq} {obj}s"
        else:
            description = f"There is a {obj}"
        object_descriptions.append(description)
   

    return object_descriptions

if __name__ == "__main__":
    # Example usage
    image_path = "chairs.jpg"  # Replace with your image path
    detected_objects = yolo_object_detection_v11(image_path)
    print("Detected objects:", detected_objects)

    object_descriptions = object_description_generator(image_path)
    print("Object descriptions:", object_descriptions)

