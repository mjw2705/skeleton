import cv2
import mediapipe as mp
import time
import math
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL


def hands(image):
    hand = mp.solutions.hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hand.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        hand_lms = [[landmark.x, landmark.y, landmark.z, landmark.visibility]
                    for landmark in results.multi_hand_landmarks[0].landmark]
        return hand_lms

    return None


def main():
    frame_w = 640
    frame_h = 480
    ptime = 0

    cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    cap_cam.set(cv2.CAP_PROP_FPS, 30)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('hand_volume.avi', fourcc, 30, (frame_w, frame_h))


    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    volBar = 200
    volPer = 0


    while cap_cam.isOpened():
        success, image = cap_cam.read()
        image = cv2.flip(image, 1)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        hand_lms = hands(image)

        if hand_lms is not None:
            x1, y1 = hand_lms[4][0], hand_lms[4][1]
            x2, y2 = hand_lms[8][0], hand_lms[8][1]
            abs_x1, abs_y1, abs_x2, abs_y2 = int(x1 * frame_w), int(y1 * frame_h), int(x2 * frame_w), int(y2 * frame_h)

            length = math.hypot(abs_x2 - abs_x1, abs_y2 - abs_y1)

            # hand range 10 ~ 120
            # vol range -65 ~ 0
            # volBar range 200 ~ 500
            vol = np.interp(length, [10, 120], [minVol, maxVol])
            volBar = np.interp(length, [10, 120], [200, 500])
            volPer = np.interp(length, [10, 120], [0, 100])

            volume.SetMasterVolumeLevel(vol, None)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(image, f'fps:{int(fps)}', (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 0, 0), 2)

        if hand_lms is not None:
            for lm in hand_lms:
                cv2.circle(image, (int(lm[0] * frame_w), int(lm[1] * frame_h)), 2, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(image, (abs_x1, abs_y1), 5, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (abs_x2, abs_y2), 5, (255, 0, 255), -1, cv2.LINE_AA)
            cv2.line(image, (abs_x1, abs_y1), (abs_x2, abs_y2), (255, 0, 255), 3)


        cv2.rectangle(image, (200, 430), (500, 460), (0, 0, 255), 3)
        cv2.rectangle(image, (200, 430), (int(volBar), 460), (0, 0, 255), cv2.FILLED)
        cv2.putText(image, f'{int(volPer)}%', (100, 453), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)

        out.write(image)
        cv2.imshow('image', image)

        if cv2.waitKey(1) == 27:
            break

    out.release()
    cap_cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()