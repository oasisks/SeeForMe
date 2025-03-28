from ultralytics import YOLO

def yolo_object_detection_v11(image_path: str) -> list:
    """
    Perform object detection using YOLO11.
    :param image_path: Path to the image file.
    :return: List of detected objects.
    """
    # Load a pre-trained YOLO11 model (e.g., YOLO11n)
    model = YOLO('yolo11s.pt')

    # Perform inference on an image
    results = model(image_path)
    return results[0].to_df()['name'].to_list()

# Example usage:
print("results", yolo_object_detection_v11("table.jpeg"))  # Provide path to your image