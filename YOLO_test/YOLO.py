from ultralytics import YOLO
from typing import List, Dict, Tuple
import numpy as np
from homography import Homog

# Initialize homography
HOMOG = Homog()

# Initialize parameters
CONF_THRESHOLD = .6
DIST_THRESHOLD = 100 # inches away from the camera

def filter_results(results) -> Dict[str, Dict[str, List|Dict]]:
    """
    Filter the results of the YOLO model to separate detected objects into left, forward, and right categories.
    :param results: Results object from YOLO model.
    """
    # left, forward, right = [], [], []
    results_dict = {"left": {"objects": [], "bounding_boxes": []},
                "forward": {"objects": [], "bounding_boxes": []},
                "right": {"objects": [], "bounding_boxes": []}}
    for result in results:
        if len(result.boxes) == 0:
            continue
        objects = result.to_df()['name'].to_list()
        boxes = result.boxes  # Boxes object for bounding box outputs
        coords = boxes.xywhn  # normalized xywh (x_center, y_center, width, height)
        xyxy_coords = boxes.xyxy # Bounding box coordinates xyxy (Top-left x, Top-left y, Bottom-right x, Bottom-right y)
        probs = boxes.conf  
        for i in range(len(objects)):
            prob = probs[i]
            coord = coords[i]
            xyxy_coord = xyxy_coords[i]
            if prob < CONF_THRESHOLD:
                continue
            # z_dist, _ = HOMOG.transformUvToXy(u=coord[0].item(), v=xyxy_coord[3].item()) # u = x_center, v = bottom-most y
            # if z_dist > DIST_THRESHOLD:
            #     continue
            x_center = coord[0]
            if x_center < 0.33:
                results_dict["left"]["objects"].append(objects[i])
                results_dict["left"]["bounding_boxes"].append(xyxy_coord.tolist())
            elif x_center > 0.66:
                results_dict["right"]["objects"].append(objects[i])
                results_dict["right"]["bounding_boxes"].append(xyxy_coord.tolist())
            else:
                results_dict["forward"]["objects"].append(objects[i])
                results_dict["forward"]["bounding_boxes"].append(xyxy_coord.tolist())
    # count the frequency of each detected object
    results_dict["left"]["objects"] = count_objects(results_dict["left"]["objects"])
    results_dict["forward"]["objects"] = count_objects(results_dict["forward"]["objects"])
    results_dict["right"]["objects"] = count_objects(results_dict["right"]["objects"])

    return results_dict

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


def yolo_object_detection_v11(img_rgb: np.ndarray) -> Dict[str, Dict[str, List|Dict]]:
    """
    Perform object detection using YOLO11.
    :param img_rgb: RGB image as a numpy array.
    :return: Dictionary of detected objects and their bounding boxes.
    """
    # Load a pre-trained YOLO11 model (e.g., YOLO11n)
    model = YOLO('yolo11x.pt')

    # Perform inference on an image
    results = model(img_rgb, verbose=False)
    results_dict = filter_results(results)

    return results_dict

def object_description_generator(detected_objects:  Dict[str, int]) -> List[str]:
    """
    Generate object descriptions based on detected objects.
    :param detected_objects: Dictionary of detected objects.
    :return: List of detected objects with descriptions.
    """
    #### TODO: use a LLM or model to generate descriptions given {"object": "2", "object1": "removed 1", "object2": "added 1"}
    # Generate descriptions for each detected object
    object_descriptions = []
    for obj, freq in detected_objects.items():
        description = f"There is {freq} {obj}."
        object_descriptions.append(description)

    return object_descriptions if len(object_descriptions) > 0 else ["No objects detected"]

if __name__ == "__main__":
    # Example usage
    image_path = "chairs.jpg"  # Replace with your image path
    detected_objects = yolo_object_detection_v11(image_path)
    print("Detected objects:", detected_objects)
    object_descriptions = object_description_generator({"chair": "2", "table": "1"})
    print("Object descriptions:", object_descriptions)


