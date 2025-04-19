import cv2
import numpy as np
import mediapipe as mp
from enum import Enum
from typing import Tuple


class FACE(Enum):
    NOSE = (0.0, 0.0, 0.0)
    CHIN = (0.0, -330.0, -65.0)
    LEFT_EYE = (-225.0, 170.0, -135.0)
    RIGHT_EYE = (225.0, 170.0, -135.0)
    LEFT_MOUTH = (-150.0, -150.0, -125.0)
    RIGHT_MOUTH = (150.0, -150.0, -125.0)


# 3D model points of a generic face model (in mm)
# https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
MODEL_POINTS = np.array([
    FACE.NOSE.value,
    FACE.CHIN.value,
    FACE.LEFT_EYE.value,
    FACE.RIGHT_EYE.value,
    FACE.LEFT_MOUTH.value,
    FACE.RIGHT_MOUTH.value
], dtype=np.float64)


# Corresponding landmark indices in MediaPipe Face Mesh
# https://github.com/google-ai-edge/mediapipe/blob/e0eef9791ebb84825197b49e09132d3643564ee2/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
class LANDMARK_IDS(Enum):
    NOSE_TIP = 1
    CHIN = 199
    LEFT_EYE = 33
    RIGHT_EYE = 263
    LEFT_MOUTH = 61
    RIGHT_MOUTH = 291


class FACE_DIRECTION(Enum):
    FORWARD = "Forward"
    LEFT = "Left"
    RIGHT = "Right"
    DOWN = "Down"
    UP = "Up"
    LEFT_UP = "Left-Up"
    LEFT_DOWN = "Left-Down"
    RIGHT_UP = "Right-Up"
    RIGHT_DOWN = "Right-Down"
    INDETERMINATE = "Indeterminate"


class Tracker:
    def __init__(self, left_yaw_threshold=-30, right_yaw_threshold=30, forward_pitch_min=140):
        """
        Initializes a tracker object

        Let x be the yaw of the head
        Let y be the pitch of the head
        Forward direction is
        left_yaw_threshold < x < right_yaw_threshold
        forward_pitch_min <= abs(pitch) < 180

        :param left_yaw_threshold: the threshold of yaw for left direction (expects negative threshold)
        :param right_yaw_threshold: the threshold of yaw for right direction (expects positive threshold)
        :param forward_pitch_min: the minimum absolute threshold for pitch to be considered as forward
        """
        self._mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self._mesh_spec = mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        self._left_yaw_threshold = left_yaw_threshold
        self._right_yaw_threshold = right_yaw_threshold
        self._forward_pitch_min = abs(forward_pitch_min)
        self._landmarks = []

    def predict_face_direction(self, img_rbg: np.ndarray) -> Tuple[FACE_DIRECTION, int, int, int]:
        """
        Predicts the face direction from the image. It expects the image to only have 1 face.
        :param img_rbg: A (h, w, 3) array where w and h can be anything
        :return: returns a FACE_DIRECTION
        """
        try:
            h, w, channels = img_rbg.shape

            if channels != 3:
                raise ValueError("The channel does not equal to 3")

            results = self._mp_face.process(img_rbg)
            self._landmarks = results.multi_face_landmarks[0]
            landmarks = self._landmarks.landmark

            # 2D image points
            image_points = []
            for LANDMARK_ID in LANDMARK_IDS:
                landmark_id = LANDMARK_ID.value
                landmark = landmarks[landmark_id]
                image_points.append((landmark.x * w, landmark.y * h))

            image_points = np.array(image_points, dtype=np.float64)

            # Camera internals
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            # Convert to Euler angles
            rot_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = np.hstack((rot_mat, translation_vector))
            _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose_mat)
            pitch, yaw, roll = euler.flatten()

            direction = FACE_DIRECTION.INDETERMINATE
            if yaw <= self._left_yaw_threshold:
                # we need to know which of the three possible left direction it could be
                if self._forward_pitch_min <= abs(pitch) < 180:
                    direction = FACE_DIRECTION.LEFT
                elif pitch < 0:
                    direction = FACE_DIRECTION.LEFT_DOWN
                else:
                    direction = FACE_DIRECTION.LEFT_UP
            elif yaw >= self._right_yaw_threshold:
                # we need to know which of the three possible right direction it could be
                if self._forward_pitch_min <= abs(pitch) < 180:
                    direction = FACE_DIRECTION.RIGHT
                elif pitch < 0:
                    direction = FACE_DIRECTION.RIGHT_DOWN
                else:
                    direction = FACE_DIRECTION.RIGHT_UP
            else:
                # if we are here, then we know that we are not turning in either left or right
                if self._forward_pitch_min <= abs(pitch) < 180:
                    direction = FACE_DIRECTION.FORWARD
                elif pitch < 0:
                    direction = FACE_DIRECTION.DOWN
                else:
                    direction = FACE_DIRECTION.UP

            return direction, pitch, yaw, roll
        except ValueError as e:
            print(e)
            return FACE_DIRECTION.INDETERMINATE, -1, -1, -1
        except TypeError as e:
            print(e)
            return FACE_DIRECTION.INDETERMINATE, -1, -1, -1

    def video_capture(self) -> None:
        """
        Calls this function to produce a live video overlay of what is being

        Press Esc to stop
        :return: None
        """
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            direction, pitch, yaw, roll = self.predict_face_direction(img_rgb)
            mp.solutions.drawing_utils.draw_landmarks(
                image=frame,
                landmark_list=self._landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mesh_spec
            )

            text = f"Pitch: {pitch:.1f}, Yaw: {yaw:.1f}, Roll: {roll:.1f}"
            cv2.putText(frame, text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, direction.value, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow("Head Pose + UV Mesh", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    tracker = Tracker(-30, 30, 165)
    tracker.video_capture()
