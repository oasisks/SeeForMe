from ultralytics import YOLO

# https://docs.ultralytics.com/usage/python/, https://github.com/ultralytics/ultralytics?tab=readme-ov-file

# Load a pre-trained YOLO11 model (e.g., YOLO11n)
model = YOLO('yolo11s.pt')

# Perform inference on an image
results = model('table.jpeg')

# Run batched inference on a list of images
results = model(["table.jpeg", "hotdog.jpeg"])  # return a list of Results objects
# Process results list

# https://docs.ultralytics.com/modes/predict/#working-with-results
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    print(result.to_df())  # print the detected objects
    print(result.to_df()['name'].to_dict())  # print the detected objects

    result.show()  # display to screen