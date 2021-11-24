import cv2
import mediapipe as mp
import time
import math
import numpy as np


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


def finger_coor(hand_lms, frame_w, frame_h):
    x1, y1 = hand_lms[4][0], hand_lms[4][1]
    x2, y2 = hand_lms[8][0], hand_lms[8][1]
    abs_x1, abs_y1, abs_x2, abs_y2 = int(x1 * frame_w), int(y1 * frame_h), int(x2 * frame_w), int(y2 * frame_h)

    return abs_x1, abs_y1, abs_x2, abs_y2


def main():
    frame_w = 512
    frame_h = 512
    ptime = 0

    # 사진
    cap_image = cv2.imread('dog.png')
    img_copy = cap_image.copy()

    # 웹캠
    cap_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap_cam.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    cap_cam.set(cv2.CAP_PROP_FPS, 30)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('hand_image.avi', fourcc, 30, (frame_w * 3, frame_h * 2))

    while cap_cam.isOpened():
        backboard = np.zeros((frame_h * 2, frame_w * 3, 3), np.uint8)

        success, image = cap_cam.read()
        image = cv2.resize(image, (frame_w, frame_h))
        image = cv2.flip(image, 1)

        if not success:
            print("Ignoring empty camera frame.")
            continue

        hand_lms = hands(image)

        if hand_lms is not None:
            abs_x1, abs_y1, abs_x2, abs_y2 = finger_coor(hand_lms, frame_w, frame_h)

            length = math.hypot(abs_x2 - abs_x1, abs_y2 - abs_y1)

            # hand range 13 ~ 120
            Per = np.interp(length, [13, 120], [0.1, 2])
            re_img = cv2.resize(cap_image, (int(frame_w * Per), int(frame_h * Per)), cv2.INTER_AREA)
        else:
            re_img = img_copy

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

        re_h, re_w, _ = re_img.shape
        backboard[:frame_h, :frame_w] = image
        backboard[:re_h, frame_w:frame_w + re_w] = re_img

        out.write(backboard)
        cv2.imshow('image', backboard)

        # cv2.imshow('image', image)
        # cv2.imshow('cap_image', re_img)

        if cv2.waitKey(1) == 27:
            break

    cap_cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()