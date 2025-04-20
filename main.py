from face_tracker.tracking import Tracker
from YOLO_test.YOLO import yolo_object_detection_v11, object_description_generator
from audio_output import text_to_speech
import cv2
import mediapipe as mp

# Global variables
scene_camera_i = 1  # Index 1 is typically the Camo webcam, but this may vary
user_camera_i = 0 # Index 0 is typically the built-in webcam

def main():
    # Initialize the face tracker
    current_objects = []
    face_tracker = Tracker(-30, 30, 165)

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
        object_descriptions = object_description_generator(scene_img_rgb)

        # if object_descriptions != current_objects:
        #     # Update the current objects list
        #     current_objects = object_descriptions
        #     # Speak the detected objects
        #     for obj in object_descriptions:
        #         text_to_speech(obj)
        print(object_descriptions)
       
        # Display the resulting frame
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

    def check_available_cameras():
        for i in range(5):  # Try camera indices 0 to 4
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Camera {i} is available")
                cap.release()

    check_available_cameras()

    # main()

    # Parse the scenario


