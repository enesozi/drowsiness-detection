import numpy as np
import dlib
from imutils import face_utils
from pygame import mixer


class DrowsinessDetector:
    LEFT_EYE_INDEX_START = 36
    LEFT_EYE_INDEX_END = 41
    RIGHT_EYE_INDEX_START = 42
    RIGHT_EYE_INDEX_END = 47

    def __init__(self, model_path, alarm_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.alarm = mixer
        self.alarm.init()
        self.alarm.music.load(alarm_path)


    def eucledian_dist(self, xx, yy):
        return np.sqrt(sum((xx - yy)**2))

    def calculate_ear(self, eye_coords):

        vert_dist1 = self.eucledian_dist(eye_coords[1], eye_coords[5])
        vert_dist2 = self.eucledian_dist(eye_coords[2], eye_coords[4])

        hor_dist = self.eucledian_dist(eye_coords[0], eye_coords[3])

        return (vert_dist1 + vert_dist2) / (2.0 * hor_dist)

    def get_eye_aspect_ratio(self, img_gray):
        rects = self.detector(img_gray, 0)

        if len(rects) > 0:
            rect = rects[0]
        else:
            return [None] * 3


        landmarks = self.predictor(img_gray, rect)
        landmarks = face_utils.shape_to_np(landmarks)


        left_eye = landmarks[
            DrowsinessDetector.LEFT_EYE_INDEX_START:DrowsinessDetector.LEFT_EYE_INDEX_END + 1]
        right_eye = landmarks[
            DrowsinessDetector.RIGHT_EYE_INDEX_START:DrowsinessDetector.RIGHT_EYE_INDEX_END + 1]
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)

        return (left_ear + right_ear) / 2.0, left_eye, right_eye

    def play_alarm(self):
        self.alarm.music.play()