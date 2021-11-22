import cv2
import numpy as np
import mediapipe as mp
import time


frame_w = 640
frame_h = 480
ptime = 0
ctime = 0

# cap = cv2.VideoCapture('WIN_20211118_21_27_13_Pro.mp4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('selfie.avi', fourcc, 30, (frame_w, frame_h))

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

BG_COLOR = (0, 0, 0)
bg_image = True

while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)

    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = selfie.process(rgb)
    image.flags.writeable = True

    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

    if bg_image is None:
        bg_images = np.zeros(image.shape, dtype=np.uint8)
        bg_images[:] = BG_COLOR
    else:
        bg_images = cv2.GaussianBlur(image, (55, 55), 0)
        # bg_images = cv2.imread('./ex3.jpg')
        # bg_images = cv2.resize(bg_image, (frame_w, frame_h))

    output_image = np.where(condition, image, bg_images)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(output_image, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)
    out.write(output_image)
    cv2.imshow('image', output_image)

    if cv2.waitKey(1) == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()