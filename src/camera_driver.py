import cv2
import numpy as np
from video import Video
from models import get_model


face_detector = cv2.CascadeClassifier(
    'venv/lib/python3.10/site-packages/cv2/data/haarcascade_frontalface_default.xml')
mouth_detector = cv2.CascadeClassifier(
    'venv/lib/python3.10/site-packages/cv2/data/haarcascade_mcs_mouth.xml')
tracker = cv2.TrackerCSRT_create()
model = get_model()


def get_mouth_rect(frame):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_height = int(frame.shape[0] * 0.35)
    min_width = int(frame.shape[1] * 0.35)
    try:
        mouths_rect = mouth_detector.detectMultiScale(converted, minNeighbors=5, scaleFactor=1.1,
                                                            minSize=(min_height, min_width))
    except cv2.error as e:
        print(e)
        return
    if not len(mouths_rect):
        return None, None, None, None
    return mouths_rect[0]


def get_face_rect(frame):
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    min_height = int(frame.shape[0] * 0.1)
    min_width = int(frame.shape[1] * 0.1)
    try:
        faces_rect = face_detector.detectMultiScale(converted, minNeighbors=3, scaleFactor=1.1,
                                                          minSize=(min_height, min_width))
    except cv2.error as e:
        print(e)
        return
    if not len(faces_rect):
        return None, None, None, None
    return faces_rect[0]

camera = cv2.VideoCapture()
camera.open(0)
assert camera.isOpened(), "Couldn't open camera (id={}) video source".format(0)

is_valid_frame = True
is_tracker_init = False
model_feed = None
while is_valid_frame:
    is_valid_frame, frame = camera.read()
    # converted = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = get_face_rect(frame)
    if x is not None:
        cropped_face_frame = frame[y:y + h, x:x + w]
        cropped_mouth_frame = cropped_face_frame[int(h/2):h, 0:w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if not is_tracker_init:
            x2, y2, w2, h2 = get_mouth_rect(cropped_mouth_frame)
            if x2 is not None:
                cv2.rectangle(frame, (x+x2, y+int(h/2)+y2), (x+x2 + w2, y+y2+int(h/2) + h2), (0, 255, 0), 2)
                # cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
                tracker.init(frame, (x+x2, y+int(h/2)+y2, w2, h2))
                is_tracker_init = True
        else:
            status_tracker, bbox = tracker.update(frame)
            x2, y2, w2, h2 = bbox
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)
            mouth_frame = frame[y2:y2 + h2, x2:x2 + w2]
            resized = cv2.resize(mouth_frame, (24, 32))
            converted = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # print(converted.shape)
            if model_feed is None:
                model_feed = converted[np.newaxis, :]
            elif model_feed.shape[0] < 12:
                model_feed = np.vstack((model_feed, converted[np.newaxis, :]))
                # print(model_feed.shape)
            else:
                words = model.predict(model_feed[np.newaxis, :])
                word_dict = {'ABOUT': words[0][0], 'DIFFERENT': words[0][1], 'EVERY': words[0][2],
                             'GOING': words[0][3], "HUMAN": words[0][4]}
                sorted_words = sorted(word_dict.items(), key=lambda z: z[1] * -1)
                print(sorted_words)
                #print("ABOUT: {}, DIFFERENT: {}, EVERY: {}, GOING: {}, HUMAN: {}".format(words[0][0], words[0][1], words[0][2], words[0][3], words[0][4]))

                model_feed = None

    cv2.imshow('Frame', frame)
    key = cv2.waitKey(4)
    if key == 'q' or key == 113:
        break

cv2.destroyAllWindows()
