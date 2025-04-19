from face_tracker.tracking import Tracker
from YOLO_test.YOLO import yolo_object_detection_v11, object_description_generator
import cv2
import mediapipe as mp

def main():
    # Initialize the face tracker
    face_tracker = Tracker(-30, 30, 165)

    # Open the webcam feed from Camo (adjust the index if needed)
    scene_camera = cv2.VideoCapture(1)  # Index 1 is typically the Camo webcam, but this may vary
    user_camera = cv2.VideoCapture(0)  # Index 0 is typically the built-in webcam
    while True:
        # Capture frame-by-frame
        ret, frame = scene_camera.read()
        ret2, frame2 = user_camera.read()

        if not ret or not ret2:
            print("Failed to capture video")
            break

        img_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        direction, pitch, yaw, roll = face_tracker.predict_face_direction(img_rgb)
        mp.solutions.drawing_utils.draw_landmarks(
                image=frame2,
                landmark_list=face_tracker._landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=face_tracker._mesh_spec
            )
       
        text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
        cv2.putText(frame2, text, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame2, direction.value, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Perform object detection on the scene camera frame
        scene_img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_descriptions = object_description_generator(scene_img_rgb)
        print(detection_descriptions)
       
        # Display the resulting frame
        cv2.imshow('Camo iPhone Camera', frame)
        cv2.imshow('User Camera', frame2)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    scene_camera.release()
    user_camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

    # Parse the scenario


