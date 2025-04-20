from ultralytics import YOLO
from typing import List, Dict
import numpy as np

def show_results(results):
    """
    Display the results of the YOLO model.
    :param results: Results object from YOLO model.
    """
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        print(result.to_df())  # print the detected objects
        print(result.to_df()['name'].to_dict())  # print the detected objects

        result.show()  


def yolo_object_detection_v11(img_rgb: np.ndarray) -> Dict[str, int]:
    """
    Perform object detection using YOLO11.
    :param img_rgb: RGB image as a numpy array.
    :return: Dictionary of detected objects.
    """
    # Load a pre-trained YOLO11 model (e.g., YOLO11n)
    model = YOLO('yolo11m.pt')

    # Perform inference on an image
    results = model(img_rgb)
    objects =  sorted(results[0].to_df()['name'].to_list()) if len(results[0]) > 0 else []
    # Find frequencies of detected objects
    object_frequencies = {}
    for obj in objects:
        if obj in object_frequencies:
            object_frequencies[obj] += 1
        else:
            object_frequencies[obj] = 1

    return object_frequencies

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

    object_descriptions = object_description_generator(image_path)
    print("Object descriptions:", object_descriptions)

