
import numpy as np
import tensorflow as tf
import mediapipe as mp


class SignBridgePredictor:

    def __init__(self, model_path, hand_model_path, class_names):

        # Load trained neural network
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        self.class_names = class_names

        # Setup MediaPipe Hand Landmarker
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=hand_model_path
            ),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3
        )

        self.detector = HandLandmarker.create_from_options(options)


    def normalize_landmarks(self, hand_landmarks):

        landmarks = []

        wrist_x = hand_landmarks[0].x
        wrist_y = hand_landmarks[0].y
        wrist_z = hand_landmarks[0].z

        for landmark in hand_landmarks:

            x = landmark.x - wrist_x
            y = landmark.y - wrist_y
            z = landmark.z - wrist_z

            landmarks.extend([x, y, z])

        landmarks = np.array(
            landmarks,
            dtype=np.float32
        )

        max_value = np.max(np.abs(landmarks))

        if max_value > 0:
            landmarks = landmarks / max_value

        return landmarks


    def close(self):
        self.detector.close()
