import face_tracker.tracking
from face_tracker.tracking import Tracker
from YOLO_test.YOLO import yolo_object_detection_v11, object_description_generator
from audio_output import text_to_speech
from transcription.transcriber import whisper_process, Transcriber
import cv2
import serial
import multiprocessing as mp
import time
import keyboard
from gemini_api import gemini_image_description

# Global variables
scene_camera_i = 1  # Index 1 is typically the Camo webcam, but this may vary
user_camera_i = 0  # Index 0 is typically the built-in webcam
frames_per_sec = 10
USE_HAPTICS = True


def scene_camera_process(cam_index, queue):
    """
    This function is run on a separate process forked from the main process

    The intent is to capture the scene and returned the objects from yolo
    :param cam_index: the cam index
    :param queue: the queue object
    :return:
    """
    capture = cv2.VideoCapture(cam_index)
    window_name = "Scene"

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Failed to capture")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        objs = yolo_object_detection_v11(img_rgb)
        queue.put(("scene", (objs, img_rgb)))

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(1 / frames_per_sec)

    capture.release()
    cv2.destroyWindow(window_name)


def user_camera_process(cam_index, queue):
    """
    This function is run on a separate process forked from the main process

    The intent is to capture the user's face direction
    :param cam_index: the cam index
    :param queue: the queue object
    :return:
    """
    capture = cv2.VideoCapture(cam_index)
    window_name = "User"
    face_tracker = Tracker(-30, 30, 165)

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Failed to capture")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        direction, pitch, yaw, roll = face_tracker.predict_face_direction(img_rgb)
        queue.put(("user", direction))

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyWindow(window_name)


def main():
    # Initialize the face tracker
    current_objects = None
    history_objects = {"left": None, "forward": None, "right": None}
    transcriber = Transcriber()
    ## Until the system announces any object, the direciton is None: after the first announcement, the direction is set to a dictionary

    # Initialize the serial port (haptics)
    if USE_HAPTICS:
        # ser = serial.Serial('/dev/tty.usbmodem1101', 9600,
        #                     timeout=1)  # Change to your port (e.g., "/dev/ttyUSB0" for Linux)
        ser = serial.Serial('COM3', 9600, timeout=1)  # Change to your port (e.g., "/dev/ttyUSB0" for Linux)
        time.sleep(2)
        
        if ser.is_open:
            print("Serial port opened successfully!")
        else:
            print("Failed to open serial port!")
        ser.flush()

    mp.set_start_method("spawn")
    result_q = mp.Queue()

    p1 = mp.Process(target=user_camera_process, args=(user_camera_i, result_q))
    p2 = mp.Process(target=scene_camera_process, args=(scene_camera_i, result_q))

    model = "base"
    mic_index = 1
    pause_threshold = 0.8
    pause_event = mp.Event()
    p3 = mp.Process(target=whisper_process, args=(result_q, pause_event, model, mic_index, pause_threshold))

    p1.start()
    p2.start()
    p3.start()
    # Open the webcam feed from Camo (adjust the index if needed)
    # user_camera = cv2.VideoCapture(user_camera_i) # Index 0 is typically the built-in webcam
    # scene_camera = cv2.VideoCapture(scene_camera_i) # Index 1 is typically the Camo webcam, but this may vary
    pause_event.set()
    flag = False
    direction = face_tracker.tracking.FACE_DIRECTION.INDETERMINATE
    while True:
        if keyboard.is_pressed("a"):
            flag = True
            pause_event.clear()
        who, data = result_q.get()
        # print(who)
        if who == "scene":
            detected_objects_dict, scene_image_rgb = data
        elif who == "whisper":
            text = data
        else:
            direction = data
        if "detected_objects_dict" in locals() and "direction" in locals():
            # print(detected_objects_dict)
            # print(direction)
            # Perform object detection on the scene camera frame
            detected_objects_left = detected_objects_dict["left"]["objects"]
            detected_objects_forward = detected_objects_dict["forward"]["objects"]
            detected_objects_right = detected_objects_dict["right"]["objects"]

            left_dir = ["left", "left-up", "left-down"]
            right_dir = ["right", "right-up", "right-down"]
            forward_dir = ["forward", "up", "down"]
            current_direction = None
            if direction.value in left_dir:
                detected_objects = detected_objects_left
                bounding_boxes = detected_objects_dict["left"]["bounding_boxes"]
                current_direction = "left"
            elif direction.value in right_dir:
                detected_objects = detected_objects_right
                bounding_boxes = detected_objects_dict["right"]["bounding_boxes"]
                current_direction = "right"
            elif direction.value in forward_dir:
                detected_objects = detected_objects_forward
                bounding_boxes = detected_objects_dict["forward"]["bounding_boxes"]
                current_direction = "forward"

            ##### CHECK FOR OBJECTS DETECTED and GENERATE THE CORRECT DESCRIPTION #####
            # 1st announcement for each direction: the objects are announced or "no objects detected"
            # 2nd announcement for each direction: the objects are announced or "no changes detected"
            if detected_objects != current_objects:  # detected objects have changed
                current_objects = detected_objects  # dict of objects with their frequency
                objects_to_announce = detected_objects.copy()

                if not current_objects: # No objects detected, clear history or start fresh
                    if history_objects[current_direction] is None:
                        objects_to_announce = None # In this case, we want to announce "no objects detected"
                        history_objects[current_direction] = {}  # start fresh so that we can detect if there's a change for next time
                    else: # announce the objects that were removed or no change detected
                        objects_to_announce = {k:f"removed {v}" for k, v in history_objects[current_direction].items()}
                        history_objects[current_direction] = {}  # clear history
                else:
                    # Update and check history
                    for obj, count in current_objects.items():
                        if history_objects[current_direction] is not None and obj in history_objects[current_direction]:  # object already announced before
                            if history_objects[current_direction][
                                obj] == count:  # nothing changed, so no need to announce
                                del objects_to_announce[obj]
                            elif history_objects[current_direction][obj] < count:  # new object detected
                                objects_to_announce[
                                    obj] = f"added {count - history_objects[current_direction][obj]}"  # only announce the new object
                                history_objects[current_direction][obj] = count
                            elif history_objects[current_direction][obj] > count:  # object disappeared
                                objects_to_announce[
                                    obj] = f"removed {history_objects[current_direction][obj] - count}"  # only announce the new object
                                history_objects[current_direction][obj] = count
                        else:
                            if history_objects[current_direction] is None:
                                history_objects[current_direction] = {}
                            history_objects[current_direction][obj] = count  # new object detected
                            objects_to_announce[obj] = f"{count}"

                # print(f"Detected objects: {objects_to_announce}")
                if current_direction == "left":
                    current_img = scene_image_rgb[:scene_image_rgb.shape[0]//3, :, :]
                elif current_direction == "right":
                    current_img = scene_image_rgb[2 * scene_image_rgb.shape[0]//3:, :, :]
                else:
                    current_img = scene_image_rgb[scene_image_rgb.shape[0]//3:2*scene_image_rgb.shape[0]//3, :, :]

                # img of the direction, dict of objects with their frequency but as strings
                object_descriptions = gemini_image_description(current_img, objects_to_announce)
                ## draw bounding boxes and labels on the scene camera frame
                # for box in bounding_boxes:
                #     x1, y1, x2, y2 = map(int, box)
                #     # Draw a rectangle (bounding box)
                #     cv2.rectangle(scene_camera_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if object_descriptions.lower().strip() == "no changes detected.":
                    continue
                if not flag:
                    text_to_speech(object_descriptions)

            # Haptics object warning loop
            if USE_HAPTICS:
                if direction.value not in left_dir and "person" in detected_objects_left:
                    ser.write(b'WARN: LEFT 200\n')
                elif direction.value in left_dir or "person" not in detected_objects_left:
                    ser.write(b'WARN: LEFT 0\n')

                if direction.value not in forward_dir and "person" in detected_objects_forward:
                    ser.write(b'WARN: FORWARD 200\n')
                elif direction.value in forward_dir or "person" not in detected_objects_forward:
                    ser.write(b'WARN: FORWARD 0\n')

                if direction.value not in right_dir and "person" in detected_objects_right:
                    ser.write(b'WARN: RIGHT 50\n')
                elif direction.value in right_dir or "person" not in detected_objects_right:
                    ser.write(b'WARN: RIGHT 0\n')
            if "text" in locals() and flag:
                flag = False
                if not text:
                    continue
                transcriber.push_user_query(text, detected_objects_dict)
                result = transcriber.get_gemini_user_response(scene_image_rgb)
                text_to_speech(result)
                pause_event.set()
                del text

            del direction, detected_objects_dict

    p1.terminate()
    p2.terminate()
    p3.terminate()
    # ser.close()


if __name__ == "__main__":
    # def check_available_cameras():
    #     for i in range(5):  # Try camera indices 0 to 4
    #         cap = cv2.VideoCapture(i)
    #         if cap.isOpened():
    #             print(f"Camera {i} is available")
    #             cap.release()

    # check_available_cameras()

    main()
    # mp.set_start_method("spawn")
    # result_q = mp.Queue()
    #
    # p1 = mp.Process(target=user_camera_process, args=(user_camera_i, result_q))
    # p2 = mp.Process(target=scene_camera_process, args=(scene_camera_i, result_q))
    # p1.start()
    # p2.start()
    # # Parse the scenario
