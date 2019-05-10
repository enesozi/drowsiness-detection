# import the necessary packages
from imutils.video import VideoStream
from threading import Thread
import imutils
import cv2
import json
from drowsiness_detector import DrowsinessDetector
import time
from pygame import mixer


with open('config.json','r') as f:
    config = json.load(f)
    MODEL_PATH = config['model_path']
    ALARM_PATH = config['alarm_path']
    EYE_AR_THRESHOLD = float(config['eye_aspect_ratio_threshold'])
    EYE_AR_MIN_FRAME_COUNT = int(config['eye_aspect_ratio_min_frame_count'])


FRAME_COUNTER = 0
ALARM_PLAY = False

detector = DrowsinessDetector(MODEL_PATH, ALARM_PATH)

vs = VideoStream(0).start()
time.sleep(1.0)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ear,left_eye,right_eye = detector.get_eye_aspect_ratio(img_gray)
    if ear is None:
        continue

    leftEyeHull = cv2.convexHull(left_eye)
    rightEyeHull = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

    if ear < EYE_AR_THRESHOLD:
        FRAME_COUNTER += 1

        if FRAME_COUNTER >= EYE_AR_MIN_FRAME_COUNT:
            
            if not ALARM_PLAY:
                ALARM_PLAY = True
                t = Thread(target=detector.play_alarm)
                t.deamon = True
                t.start()

            cv2.putText(frame, "WAKE UP!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        FRAME_COUNTER = 0
        ALARM_PLAY = False


    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
