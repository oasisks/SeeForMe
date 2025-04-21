from face_tracker.tracking import Tracker
from YOLO_test.YOLO import yolo_object_detection_v11, object_description_generator
from audio_output import text_to_speech
import cv2
import mediapipe as mp
import serial
import time

# Global variables
scene_camera_i = 1  # Index 1 is typically the Camo webcam, but this may vary
user_camera_i = 0 # Index 0 is typically the built-in webcam

def main():
    # Initialize the face tracker
    current_objects = None
    face_tracker = Tracker(-30, 30, 165)

    # Initialize the serial port (haptics)
    ser = serial.Serial('/dev/tty.usbmodem1101', 9600, timeout=1)  # Change to your port (e.g., "/dev/ttyUSB0" for Linux)
    # ser = serial.Serial('COM3', 9600, timeout=1)  # Change to your port (e.g., "/dev/ttyUSB0" for Linux)
    time.sleep(2)

    if ser.is_open:
        print("Serial port opened successfully!")
    else:
        print("Failed to open serial port!")
    ser.flush()

    # Open the webcam feed from Camo (adjust the index if needed)
    user_camera = cv2.VideoCapture(user_camera_i)  # Index 0 is typically the built-in webcam
    scene_camera = cv2.VideoCapture(scene_camera_i)  # Index 1 is typically the Camo webcam, but this may vary
    while True:
        # Capture frame-by-frame
        ret, scene_camera_frame = scene_camera.read()
        ret2, user_camera_frame = user_camera.read()

        if not ret or not ret2:
            print("Failed to capture video")
            break
        
        # Draw the face mesh on the user camera frame
        user_img_rgb = cv2.cvtColor(user_camera_frame, cv2.COLOR_BGR2RGB)
        direction, pitch, yaw, roll = face_tracker.predict_face_direction(user_img_rgb)
        mp.solutions.drawing_utils.draw_landmarks(
                image=user_camera_frame,
                landmark_list=face_tracker._landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=face_tracker._mesh_spec
            )
       
        text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
        cv2.putText(user_camera_frame, text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(user_camera_frame, direction.value, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Perform object detection on the scene camera frame
        scene_img_rgb = cv2.cvtColor(scene_camera_frame, cv2.COLOR_BGR2RGB)
        detected_objects_dict = yolo_object_detection_v11(scene_img_rgb)
        detected_objects_left = detected_objects_dict["left"]["objects"]
        detected_objects_forward = detected_objects_dict["forward"]["objects"]
        detected_objects_right = detected_objects_dict["right"]["objects"]

        left_dir = ["Left", "Left-Up", "Left-Down"]
        right_dir = ["Right", "Right-Up", "Right-Down"]
        forward_dir = ["Forward", "Up", "Down"]
        if direction.value in left_dir:
            detected_objects = detected_objects_left
            bounding_boxes = detected_objects_dict["left"]["bounding_boxes"]
        elif direction.value in right_dir:
            detected_objects = detected_objects_right
            bounding_boxes = detected_objects_dict["right"]["bounding_boxes"]
        elif direction.value in forward_dir:
            detected_objects = detected_objects_forward
            bounding_boxes = detected_objects_dict["forward"]["bounding_boxes"]

        if detected_objects != current_objects:
            current_objects = detected_objects
            object_descriptions = object_description_generator(detected_objects)
            # draw bounding boxes and labels on the scene camera frame
            for box in bounding_boxes:
                x1, y1, x2, y2 = map(int, box)
                # Draw a rectangle (bounding box)
                cv2.rectangle(scene_camera_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text_to_speech(object_descriptions)       

        # Haptics object warning loop
        if direction.value not in left_dir and "sports ball" in detected_objects_left:
            ser.write(b'WARN: LEFT ON\n')
        elif direction.value in left_dir or "sports ball" not in detected_objects_left:
            ser.write(b'WARN: LEFT OFF\n')

        if direction.value not in forward_dir and "sports ball" in detected_objects_forward:
            ser.write(b'WARN: FORWARD ON\n')
        elif direction.value in forward_dir or "sports ball" not in detected_objects_forward:
            ser.write(b'WARN: FORWARD OFF\n')

        if direction.value not in right_dir  and "sports ball" in detected_objects_right:
            ser.write(b'WARN: RIGHT ON\n')
        elif direction.value in right_dir or "sports ball" not in detected_objects_right:
            ser.write(b'WARN: RIGHT OFF\n')

        # # Display the resulting frame
        cv2.imshow('Camo iPhone Camera', scene_camera_frame)
        cv2.imshow('User Camera', user_camera_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        

    # Release the camera and close all OpenCV windows
    scene_camera.release()
    user_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # def check_available_cameras():
    #     for i in range(5):  # Try camera indices 0 to 4
    #         cap = cv2.VideoCapture(i)
    #         if cap.isOpened():
    #             print(f"Camera {i} is available")
    #             cap.release()

    # check_available_cameras()

    main()

    # Parse the scenario


