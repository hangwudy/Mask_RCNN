# coding: utf-8

import cv2
import numpy as np
from visualize_cv2_car_door import model, display_instances, class_names

# take the frames from video
capture = cv2.VideoCapture('IMG_5634.MOV')

size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
# Write the video file
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter('videofile_masked2.avi', codec, 60.0, size)

while capture.isOpened():
    # bool, and frame value
    ret, frame = capture.read()
    if ret:
        # add mask to frame
        results = model.detect([frame], verbose=0)
        r = results[0]
        frame = display_instances(
            frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
        )
        output.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
        

capture.release()
output.release()
cv2.destroyAllWindows()